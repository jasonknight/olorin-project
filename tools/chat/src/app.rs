//! Application state and logic

use anyhow::Result;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::Duration;
use tui_textarea::TextArea;

use crate::db::ChatDb;
use crate::kafka::KafkaProducer;
use crate::message::{ChatMessage, DisplayMessage};

/// Messages sent from background threads to the main app
#[derive(Debug)]
pub enum AppEvent {
    /// New chat messages from the database
    NewChatMessages(Vec<ChatMessage>),
    /// Updated chat messages (streaming updates to existing messages)
    UpdatedChatMessages(Vec<ChatMessage>),
    /// Error occurred in background thread
    Error(String),
}

/// Application state
pub struct App<'a> {
    /// All messages to display (chat + system)
    pub messages: Vec<DisplayMessage>,
    /// Input text area
    pub input: TextArea<'a>,
    /// Scroll offset for the chat display (lines from bottom)
    pub scroll_offset: usize,
    /// Whether to auto-scroll to bottom when new messages arrive
    pub auto_scroll: bool,
    /// Whether the app should quit
    pub should_quit: bool,
    /// Kafka producer for sending prompts
    producer: Option<KafkaProducer>,
    /// Channel receiver for background events
    event_rx: Receiver<AppEvent>,
    /// Status message
    pub status: String,
    /// Whether we're in sending state
    pub is_sending: bool,
}

impl<'a> App<'a> {
    pub fn new(
        chat_db_path: PathBuf,
        bootstrap_servers: &str,
    ) -> Result<Self> {
        // Create channel for background events
        let (tx, rx) = mpsc::channel();

        // Start database monitoring thread
        Self::start_db_monitor(tx, chat_db_path);

        // Create Kafka producer
        let producer = match KafkaProducer::new(bootstrap_servers, "ai_in") {
            Ok(p) => Some(p),
            Err(e) => {
                eprintln!("Warning: Could not create Kafka producer: {}", e);
                None
            }
        };

        // Create input text area
        let mut input = TextArea::default();
        input.set_cursor_line_style(ratatui::style::Style::default());
        input.set_placeholder_text("Type your message... (Enter to send, Esc to quit)");

        Ok(Self {
            messages: Vec::new(),
            input,
            scroll_offset: 0,
            auto_scroll: true,
            should_quit: false,
            producer,
            event_rx: rx,
            status: String::from("Ready"),
            is_sending: false,
        })
    }

    /// Start the database monitoring background thread
    fn start_db_monitor(tx: Sender<AppEvent>, chat_db_path: PathBuf) {
        thread::spawn(move || {
            let mut chat_db = ChatDb::new(chat_db_path);

            // Initial load of chat messages
            match chat_db.get_all_messages() {
                Ok(messages) if !messages.is_empty() => {
                    let _ = tx.send(AppEvent::NewChatMessages(messages));
                }
                Err(e) => {
                    let _ = tx.send(AppEvent::Error(format!("Chat DB error: {}", e)));
                }
                _ => {}
            }

            // Poll loop
            loop {
                thread::sleep(Duration::from_millis(150));

                // Check for new and updated chat messages
                match chat_db.get_new_and_updated_messages() {
                    Ok(result) => {
                        // Send new messages
                        if !result.new_messages.is_empty() {
                            if tx.send(AppEvent::NewChatMessages(result.new_messages)).is_err() {
                                break; // Channel closed, exit thread
                            }
                        }
                        // Send updated messages (streaming updates)
                        if !result.updated_messages.is_empty() {
                            if tx.send(AppEvent::UpdatedChatMessages(result.updated_messages)).is_err() {
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(AppEvent::Error(format!("Chat DB error: {}", e)));
                    }
                }
            }
        });
    }

    /// Process events from background threads
    pub fn process_events(&mut self) {
        while let Ok(event) = self.event_rx.try_recv() {
            match event {
                AppEvent::NewChatMessages(messages) => {
                    for msg in messages {
                        self.messages.push(DisplayMessage::Chat(msg));
                    }
                    // Sort messages by timestamp
                    self.messages.sort_by_key(|m| m.created_at());
                    // Only auto-scroll if enabled
                    if self.auto_scroll {
                        self.scroll_offset = 0;
                    }
                }
                AppEvent::UpdatedChatMessages(updated_messages) => {
                    // Update existing messages in place (for streaming updates)
                    for updated_msg in updated_messages {
                        // Find the existing message by ID and replace it
                        for existing in &mut self.messages {
                            if let DisplayMessage::Chat(ref mut chat_msg) = existing {
                                if chat_msg.id == updated_msg.id {
                                    // Update the content and updated_at timestamp
                                    chat_msg.content = updated_msg.content.clone();
                                    chat_msg.updated_at = updated_msg.updated_at;
                                    break;
                                }
                            }
                        }
                    }
                    // Auto-scroll to show updated content
                    if self.auto_scroll {
                        self.scroll_offset = 0;
                    }
                }
                AppEvent::Error(err) => {
                    self.status = format!("Error: {}", err);
                }
            }
        }
    }

    /// Send the current input as a prompt
    pub async fn send_message(&mut self) {
        let text: String = self.input.lines().join("\n").trim().to_string();

        if text.is_empty() {
            return;
        }

        if let Some(ref producer) = self.producer {
            self.is_sending = true;
            self.status = String::from("Sending...");

            match producer.send_prompt(&text).await {
                Ok(()) => {
                    self.status = String::from("Sent!");
                    // Clear the input
                    self.input = TextArea::default();
                    self.input.set_cursor_line_style(ratatui::style::Style::default());
                    self.input.set_placeholder_text(
                        "Type your message... (Enter to send, Esc to quit)",
                    );
                }
                Err(e) => {
                    self.status = format!("Send failed: {}", e);
                }
            }

            self.is_sending = false;
        } else {
            self.status = String::from("Kafka not connected");
        }
    }

    /// Scroll the chat display up (disables auto-scroll)
    pub fn scroll_up(&mut self, amount: usize) {
        self.scroll_offset = self.scroll_offset.saturating_add(amount);
        self.auto_scroll = false;
    }

    /// Scroll the chat display down (re-enables auto-scroll at bottom)
    pub fn scroll_down(&mut self, amount: usize) {
        self.scroll_offset = self.scroll_offset.saturating_sub(amount);
        // Re-enable auto-scroll when we reach the bottom
        if self.scroll_offset == 0 {
            self.auto_scroll = true;
        }
    }

    /// Scroll to the bottom of the chat (re-enables auto-scroll)
    pub fn scroll_to_bottom(&mut self) {
        self.scroll_offset = 0;
        self.auto_scroll = true;
    }

    /// Scroll to the top of the chat (disables auto-scroll)
    pub fn scroll_to_top(&mut self) {
        // Use a large value; UI will clamp it
        self.scroll_offset = usize::MAX / 2;
        self.auto_scroll = false;
    }

    /// Get the number of messages
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }
}
