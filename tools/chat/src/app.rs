//! Application state and logic

use anyhow::Result;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::Duration;
use tiktoken_rs::cl100k_base;
use tui_textarea::TextArea;

use crate::api::{self, SlashCommand};
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
    /// Control API base URL
    control_api_url: String,
    /// Available slash commands from control API
    pub slash_commands: Vec<SlashCommand>,
    /// Current autocomplete ghost text (portion to append)
    pub completion: Option<String>,
    /// Total token count for all messages in the chat
    pub token_count: usize,
}

impl<'a> App<'a> {
    pub fn new(
        chat_db_path: PathBuf,
        bootstrap_servers: &str,
        control_api_url: &str,
        slash_commands: Vec<SlashCommand>,
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
            control_api_url: control_api_url.to_string(),
            slash_commands,
            completion: None,
            token_count: 0,
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
                        if !result.new_messages.is_empty()
                            && tx
                                .send(AppEvent::NewChatMessages(result.new_messages))
                                .is_err()
                        {
                            break; // Channel closed, exit thread
                        }
                        // Send updated messages (streaming updates)
                        if !result.updated_messages.is_empty()
                            && tx
                                .send(AppEvent::UpdatedChatMessages(result.updated_messages))
                                .is_err()
                        {
                            break;
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
        let mut messages_changed = false;

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
                    messages_changed = true;
                }
                AppEvent::UpdatedChatMessages(updated_messages) => {
                    // Update existing messages in place (for streaming updates)
                    for updated_msg in updated_messages {
                        // Find the existing message by ID and replace it
                        for existing in &mut self.messages {
                            if let DisplayMessage::Chat(chat_msg) = existing {
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
                    messages_changed = true;
                }
                AppEvent::Error(err) => {
                    self.status = format!("Error: {}", err);
                }
            }
        }

        // Recalculate token count if messages changed
        if messages_changed {
            self.recalculate_token_count();
        }
    }

    /// Recalculate the total token count for all messages
    fn recalculate_token_count(&mut self) {
        let bpe = match cl100k_base() {
            Ok(bpe) => bpe,
            Err(_) => {
                self.token_count = 0;
                return;
            }
        };

        let mut total = 0;
        for msg in &self.messages {
            if let DisplayMessage::Chat(chat_msg) = msg {
                // Count tokens in the message content
                total += bpe.encode_with_special_tokens(&chat_msg.content).len();
                // Add overhead for role (roughly 4 tokens per message for role/formatting)
                total += 4;
            }
        }
        self.token_count = total;
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
                    self.input
                        .set_cursor_line_style(ratatui::style::Style::default());
                    self.input
                        .set_placeholder_text("Type your message... (Enter to send, Esc to quit)");
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

    /// Find completion for partial slash command input
    ///
    /// Returns the portion of the command to append (ghost text),
    /// or None if no completion is available.
    pub fn find_completion(&self, input: &str) -> Option<String> {
        if !input.starts_with('/') {
            return None;
        }

        self.slash_commands
            .iter()
            .find(|cmd| cmd.slash_command.starts_with(input) && cmd.slash_command != input)
            .map(|cmd| cmd.slash_command[input.len()..].to_string())
    }

    /// Update the completion based on current input
    pub fn update_completion(&mut self) {
        let text: String = self.input.lines().join("\n");
        // Only show completion for single-line input starting with /
        if text.starts_with('/') && !text.contains('\n') {
            self.completion = self.find_completion(&text);
        } else {
            self.completion = None;
        }
    }

    /// Accept the current completion (insert ghost text)
    pub fn accept_completion(&mut self) {
        if let Some(completion) = self.completion.take() {
            for c in completion.chars() {
                self.input.insert_char(c);
            }
        }
    }

    /// Execute a slash command via the control API
    pub async fn execute_slash_command(&mut self, text: &str) {
        self.is_sending = true;
        self.status = String::from("Executing command...");

        // Extract command name for special handling
        let command_name = text.split_whitespace().next().unwrap_or("");

        match api::execute_command(&self.control_api_url, text).await {
            Ok(response) => {
                if response.success {
                    if let Some(result) = response.result {
                        if let Some(message) = result.get("message").and_then(|m| m.as_str()) {
                            self.status = message.to_string();
                        } else {
                            self.status = String::from("Command executed successfully");
                        }
                    } else {
                        self.status = String::from("Command executed successfully");
                    }

                    // Clear the display when /clear command succeeds
                    if command_name == "/clear" {
                        self.messages.clear();
                        self.token_count = 0;
                    }
                } else {
                    self.status = format!(
                        "Command failed: {}",
                        response
                            .error
                            .unwrap_or_else(|| "Unknown error".to_string())
                    );
                }
                // Clear the input on success
                self.input = TextArea::default();
                self.input
                    .set_cursor_line_style(ratatui::style::Style::default());
                self.input
                    .set_placeholder_text("Type your message... (Enter to send, Esc to quit)");
                self.completion = None;
            }
            Err(e) => {
                self.status = format!("Command error: {}", e);
                // Still clear the input on error
                self.input = TextArea::default();
                self.input
                    .set_cursor_line_style(ratatui::style::Style::default());
                self.input
                    .set_placeholder_text("Type your message... (Enter to send, Esc to quit)");
                self.completion = None;
            }
        }

        self.is_sending = false;
    }
}
