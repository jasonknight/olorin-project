//! Application state and logic

use anyhow::Result;
use olorin_state::State;
use std::cell::Cell;
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

/// Active tab in the TUI
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ActiveTab {
    #[default]
    Chat,
    State,
    Search,
}

/// Cached state data for display
#[derive(Debug, Clone, Default)]
pub struct StateDisplay {
    pub entries: Vec<(String, String, String)>, // (key, value, type)
    pub last_refresh: Option<chrono::DateTime<chrono::Local>>,
    pub scroll_offset: usize,
    /// Instant of last refresh for periodic update checks
    #[allow(dead_code)]
    last_refresh_instant: Option<std::time::Instant>,
}

/// A search result from ChromaDB
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub text: String,
    pub source: Option<String>,
    pub distance: Option<f64>,
    pub is_in_context: bool,
}

/// Focus state for search tab
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchFocus {
    #[default]
    Input,
    Results,
}

/// Focus state for manual entry modal fields
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ManualEntryFocus {
    #[default]
    Text,
    Source,
}

/// Search mode: semantic (embeddings) or source (filename match)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchMode {
    #[default]
    Semantic,
    Source,
}

impl SearchMode {
    /// Get display name for the mode
    pub fn as_str(&self) -> &'static str {
        match self {
            SearchMode::Semantic => "Semantic",
            SearchMode::Source => "Source",
        }
    }

    /// Toggle to the other mode
    pub fn toggle(&self) -> Self {
        match self {
            SearchMode::Semantic => SearchMode::Source,
            SearchMode::Source => SearchMode::Semantic,
        }
    }
}

/// State for the search tab
#[derive(Debug, Clone, Default)]
pub struct SearchState {
    /// Search query input
    pub query: String,
    /// Current search results
    pub results: Vec<SearchResult>,
    /// Selected result index
    pub selected_index: usize,
    /// Whether showing document modal
    pub showing_modal: bool,
    /// Whether showing help modal
    pub showing_help: bool,
    /// Current focus (input or results)
    pub focus: SearchFocus,
    /// Search mode (semantic or source)
    pub mode: SearchMode,
    /// Loading state
    pub is_loading: bool,
    /// Error message if any
    pub error: Option<String>,
    /// Scroll offset for results list
    pub scroll_offset: usize,
    /// IDs of documents in context
    pub context_ids: std::collections::HashSet<String>,
    /// Token count for documents in context
    pub context_token_count: usize,
    /// Whether showing manual entry modal
    pub showing_manual_entry: bool,
    /// Manual entry text content
    pub manual_entry_text: String,
    /// Manual entry source (defaults to "User Context")
    pub manual_entry_source: String,
    /// Current focus in manual entry modal
    pub manual_entry_focus: ManualEntryFocus,
    /// Whether manual entry is being submitted (loading)
    pub manual_entry_loading: bool,
    /// Cursor position in text field (character index)
    pub manual_entry_text_cursor: usize,
    /// Cursor position in source field (character index)
    pub manual_entry_source_cursor: usize,
    /// Scroll offset for text field (lines from top, uses Cell for interior mutability during render)
    pub manual_entry_scroll_offset: std::cell::Cell<usize>,
}

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
    /// Scroll offset for the input area (lines from top, uses Cell for interior mutability during render)
    pub input_scroll_offset: Cell<usize>,
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
    /// Active tab in the TUI
    pub active_tab: ActiveTab,
    /// Cached state data for display
    pub state_display: StateDisplay,
    /// State for the search tab
    pub search_state: SearchState,
    /// Search tool URL
    search_tool_url: String,
    /// Context DB path
    context_db_path: std::path::PathBuf,
    /// Whether showing the quit confirmation modal
    pub showing_quit_modal: bool,
}

impl<'a> App<'a> {
    pub fn new(
        chat_db_path: PathBuf,
        bootstrap_servers: &str,
        control_api_url: &str,
        slash_commands: Vec<SlashCommand>,
        search_tool_url: &str,
        context_db_path: PathBuf,
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
            input_scroll_offset: Cell::new(0),
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
            active_tab: ActiveTab::default(),
            state_display: StateDisplay::default(),
            search_state: SearchState::default(),
            search_tool_url: search_tool_url.to_string(),
            context_db_path,
            showing_quit_modal: false,
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

        // Periodic state refresh when on State tab (every 1 second)
        if self.active_tab == ActiveTab::State {
            let should_refresh = match self.state_display.last_refresh_instant {
                Some(instant) => instant.elapsed() >= Duration::from_secs(1),
                None => true, // Never refreshed yet
            };
            if should_refresh {
                self.refresh_state_silent();
            }
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
                        // Check if handler returned an error in its result
                        if let Some(error) = result.get("error").and_then(|e| e.as_str()) {
                            self.status = format!("Error: {}", error);
                        } else if let Some(message) = result.get("message").and_then(|m| m.as_str())
                        {
                            self.status = message.to_string();
                            // Clear the display when /clear command succeeds
                            if command_name == "/clear" {
                                self.messages.clear();
                                self.token_count = 0;
                            }
                        } else {
                            self.status = String::from("Command executed successfully");
                            // Clear the display when /clear command succeeds
                            if command_name == "/clear" {
                                self.messages.clear();
                                self.token_count = 0;
                            }
                        }
                    } else {
                        self.status = String::from("Command executed successfully");
                        // Clear the display when /clear command succeeds
                        if command_name == "/clear" {
                            self.messages.clear();
                            self.token_count = 0;
                        }
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

    /// Switch to the next tab
    pub fn next_tab(&mut self) {
        self.active_tab = match self.active_tab {
            ActiveTab::Chat => {
                // Refresh state when switching to State tab
                self.refresh_state();
                ActiveTab::State
            }
            ActiveTab::State => {
                // Load context when switching to Search tab
                self.load_context_ids();
                if self.search_state.query.is_empty() {
                    self.load_context_as_results();
                }
                ActiveTab::Search
            }
            ActiveTab::Search => ActiveTab::Chat,
        };
    }

    /// Refresh the state display from olorin-state database
    pub fn refresh_state(&mut self) {
        match State::new(None) {
            Ok(state) => {
                match state.keys(None) {
                    Ok(keys) => {
                        let mut entries = Vec::new();
                        for key in keys {
                            // Get value and type for each key
                            let value_str = if let Ok(Some(v)) = state.get_string(&key) {
                                v
                            } else if let Ok(Some(v)) = state.get_int(&key) {
                                v.to_string()
                            } else if let Ok(v) = state.get_bool(&key) {
                                v.to_string()
                            } else if let Ok(Some(v)) = state.get_float(&key) {
                                v.to_string()
                            } else if let Ok(Some(v)) = state.get_json(&key) {
                                v.to_string()
                            } else {
                                "(unknown)".to_string()
                            };

                            // Get the type
                            let type_str = match state.get_type(&key) {
                                Ok(Some(t)) => format!("{:?}", t),
                                _ => "?".to_string(),
                            };

                            entries.push((key, value_str, type_str));
                        }
                        // Sort entries by key for consistent display
                        entries.sort_by(|a, b| a.0.cmp(&b.0));
                        self.state_display.entries = entries;
                        self.state_display.last_refresh = Some(chrono::Local::now());
                        self.state_display.last_refresh_instant = Some(std::time::Instant::now());
                        self.status = format!(
                            "State refreshed: {} entries",
                            self.state_display.entries.len()
                        );
                    }
                    Err(e) => {
                        self.status = format!("Failed to list state keys: {}", e);
                    }
                }
            }
            Err(e) => {
                self.status = format!("Failed to open state db: {}", e);
            }
        }
    }

    /// Refresh the state display silently (without updating status message)
    /// Used for periodic background refresh.
    fn refresh_state_silent(&mut self) {
        if let Ok(state) = State::new(None) {
            if let Ok(keys) = state.keys(None) {
                let mut entries = Vec::new();
                for key in keys {
                    let value_str = if let Ok(Some(v)) = state.get_string(&key) {
                        v
                    } else if let Ok(Some(v)) = state.get_int(&key) {
                        v.to_string()
                    } else if let Ok(v) = state.get_bool(&key) {
                        v.to_string()
                    } else if let Ok(Some(v)) = state.get_float(&key) {
                        v.to_string()
                    } else if let Ok(Some(v)) = state.get_json(&key) {
                        v.to_string()
                    } else {
                        "(unknown)".to_string()
                    };

                    let type_str = match state.get_type(&key) {
                        Ok(Some(t)) => format!("{:?}", t),
                        _ => "?".to_string(),
                    };

                    entries.push((key, value_str, type_str));
                }
                entries.sort_by(|a, b| a.0.cmp(&b.0));
                self.state_display.entries = entries;
                self.state_display.last_refresh = Some(chrono::Local::now());
                self.state_display.last_refresh_instant = Some(std::time::Instant::now());
            }
        }
    }

    /// Scroll the state display up
    pub fn scroll_state_up(&mut self, amount: usize) {
        self.state_display.scroll_offset = self.state_display.scroll_offset.saturating_add(amount);
    }

    /// Scroll the state display down
    pub fn scroll_state_down(&mut self, amount: usize) {
        self.state_display.scroll_offset = self.state_display.scroll_offset.saturating_sub(amount);
    }

    // ==================== Search Tab Methods ====================

    /// Load context IDs from the context database
    pub fn load_context_ids(&mut self) {
        self.search_state.context_ids.clear();
        self.search_state.context_token_count = 0;

        if !self.context_db_path.exists() {
            return;
        }

        if let Ok(conn) = rusqlite::Connection::open(&self.context_db_path) {
            // Ensure table exists
            let _ = conn.execute(
                "CREATE TABLE IF NOT EXISTS context_documents (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    source TEXT,
                    added_at TEXT DEFAULT CURRENT_TIMESTAMP
                )",
                [],
            );

            if let Ok(mut stmt) = conn.prepare("SELECT id, text FROM context_documents") {
                if let Ok(rows) = stmt.query_map([], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                }) {
                    let bpe = cl100k_base().ok();
                    for row in rows.flatten() {
                        let (id, text) = row;
                        self.search_state.context_ids.insert(id);
                        // Count tokens for this document
                        if let Some(ref bpe) = bpe {
                            self.search_state.context_token_count +=
                                bpe.encode_with_special_tokens(&text).len();
                        }
                    }
                }
            }
        }
    }

    /// Recalculate context token count from database
    #[allow(dead_code)]
    fn recalculate_context_tokens(&mut self) {
        self.search_state.context_token_count = 0;

        if !self.context_db_path.exists() {
            return;
        }

        if let Ok(conn) = rusqlite::Connection::open(&self.context_db_path) {
            if let Ok(mut stmt) = conn.prepare("SELECT text FROM context_documents") {
                if let Ok(rows) = stmt.query_map([], |row| row.get::<_, String>(0)) {
                    if let Ok(bpe) = cl100k_base() {
                        for text in rows.flatten() {
                            self.search_state.context_token_count +=
                                bpe.encode_with_special_tokens(&text).len();
                        }
                    }
                }
            }
        }
    }

    /// Load context documents as results (for empty search)
    pub fn load_context_as_results(&mut self) {
        self.search_state.results.clear();
        self.search_state.selected_index = 0;
        self.search_state.scroll_offset = 0;

        if !self.context_db_path.exists() {
            return;
        }

        if let Ok(conn) = rusqlite::Connection::open(&self.context_db_path) {
            if let Ok(mut stmt) = conn
                .prepare("SELECT id, text, source FROM context_documents ORDER BY added_at DESC")
            {
                if let Ok(rows) = stmt.query_map([], |row| {
                    Ok(SearchResult {
                        id: row.get(0)?,
                        text: row.get(1)?,
                        source: row.get(2)?,
                        distance: None,
                        is_in_context: true,
                    })
                }) {
                    self.search_state.results = rows.flatten().collect();
                }
            }
        }
    }

    /// Execute a search against the search tool
    pub async fn execute_search(&mut self) {
        let query = self.search_state.query.trim().to_string();

        if query.is_empty() {
            self.load_context_as_results();
            self.status = format!(
                "Showing {} context documents",
                self.search_state.results.len()
            );
            return;
        }

        self.search_state.is_loading = true;
        self.search_state.error = None;
        let mode_str = match self.search_state.mode {
            SearchMode::Semantic => "semantic",
            SearchMode::Source => "source",
        };
        self.status = format!("Searching ({})...", mode_str);

        let url = format!("{}/call", self.search_tool_url);
        let client = reqwest::Client::new();

        match client
            .post(&url)
            .json(&serde_json::json!({
                "query": query,
                "mode": mode_str,
                "per_page": 50
            }))
            .timeout(std::time::Duration::from_secs(30))
            .send()
            .await
        {
            Ok(response) => {
                if response.status().is_success() {
                    match response.json::<serde_json::Value>().await {
                        Ok(data) => {
                            if data
                                .get("success")
                                .and_then(|v| v.as_bool())
                                .unwrap_or(false)
                            {
                                let results = data
                                    .get("result")
                                    .and_then(|r| r.get("results"))
                                    .and_then(|r| r.as_array())
                                    .map(|arr| {
                                        arr.iter()
                                            .map(|item| {
                                                let id = item
                                                    .get("id")
                                                    .and_then(|v| v.as_str())
                                                    .unwrap_or("")
                                                    .to_string();
                                                SearchResult {
                                                    id: id.clone(),
                                                    text: item
                                                        .get("text")
                                                        .and_then(|v| v.as_str())
                                                        .unwrap_or("")
                                                        .to_string(),
                                                    source: item
                                                        .get("metadata")
                                                        .and_then(|m| m.get("source"))
                                                        .and_then(|v| v.as_str())
                                                        .map(|s| s.to_string()),
                                                    distance: item
                                                        .get("distance")
                                                        .and_then(|v| v.as_f64()),
                                                    is_in_context: self
                                                        .search_state
                                                        .context_ids
                                                        .contains(&id),
                                                }
                                            })
                                            .collect::<Vec<_>>()
                                    })
                                    .unwrap_or_default();

                                let count = results.len();
                                self.search_state.results = results;
                                self.search_state.selected_index = 0;
                                self.search_state.scroll_offset = 0;
                                self.status = format!("Found {} results", count);
                            } else {
                                let error = data
                                    .get("error")
                                    .and_then(|e| e.get("message"))
                                    .and_then(|m| m.as_str())
                                    .unwrap_or("Unknown error");
                                self.search_state.error = Some(error.to_string());
                                self.status = format!("Search error: {}", error);
                            }
                        }
                        Err(e) => {
                            self.search_state.error = Some(format!("JSON parse error: {}", e));
                            self.status = format!("Error: {}", e);
                        }
                    }
                } else {
                    let status = response.status();
                    self.search_state.error = Some(format!("HTTP {}", status));
                    self.status = format!("Search failed: HTTP {}", status);
                }
            }
            Err(e) => {
                self.search_state.error = Some(format!("Request error: {}", e));
                self.status = format!("Search error: {}", e);
            }
        }

        self.search_state.is_loading = false;
    }

    /// Add the currently selected document to context
    pub fn add_to_context(&mut self) {
        if self.search_state.results.is_empty() {
            return;
        }

        let idx = self.search_state.selected_index;
        if idx >= self.search_state.results.len() {
            return;
        }

        let result = &self.search_state.results[idx];
        if result.is_in_context {
            self.status = String::from("Already in context");
            return;
        }

        // Calculate tokens for this document before adding
        let doc_tokens = cl100k_base()
            .map(|bpe| bpe.encode_with_special_tokens(&result.text).len())
            .unwrap_or(0);

        if let Ok(conn) = rusqlite::Connection::open(&self.context_db_path) {
            // Ensure table exists
            let _ = conn.execute(
                "CREATE TABLE IF NOT EXISTS context_documents (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    source TEXT,
                    added_at TEXT DEFAULT CURRENT_TIMESTAMP
                )",
                [],
            );

            match conn.execute(
                "INSERT OR IGNORE INTO context_documents (id, text, source) VALUES (?1, ?2, ?3)",
                [
                    &result.id,
                    &result.text,
                    &result.source.clone().unwrap_or_default(),
                ],
            ) {
                Ok(_) => {
                    self.search_state.context_ids.insert(result.id.clone());
                    self.search_state.context_token_count += doc_tokens;
                    // Update the result to show it's in context
                    if let Some(r) = self.search_state.results.get_mut(idx) {
                        r.is_in_context = true;
                    }
                    self.status = String::from("Added to context");
                }
                Err(e) => {
                    self.status = format!("Failed to add: {}", e);
                }
            }
        } else {
            self.status = String::from("Failed to open context database");
        }
    }

    /// Remove the currently selected document from context
    pub fn remove_from_context(&mut self) {
        if self.search_state.results.is_empty() {
            return;
        }

        let idx = self.search_state.selected_index;
        if idx >= self.search_state.results.len() {
            return;
        }

        let result = &self.search_state.results[idx];
        if !result.is_in_context {
            self.status = String::from("Not in context");
            return;
        }

        // Calculate tokens for this document before removing
        let doc_tokens = cl100k_base()
            .map(|bpe| bpe.encode_with_special_tokens(&result.text).len())
            .unwrap_or(0);

        if let Ok(conn) = rusqlite::Connection::open(&self.context_db_path) {
            match conn.execute("DELETE FROM context_documents WHERE id = ?1", [&result.id]) {
                Ok(_) => {
                    self.search_state.context_ids.remove(&result.id);
                    self.search_state.context_token_count = self
                        .search_state
                        .context_token_count
                        .saturating_sub(doc_tokens);
                    // Update the result to show it's not in context
                    if let Some(r) = self.search_state.results.get_mut(idx) {
                        r.is_in_context = false;
                    }
                    self.status = String::from("Removed from context");
                }
                Err(e) => {
                    self.status = format!("Failed to remove: {}", e);
                }
            }
        } else {
            self.status = String::from("Failed to open context database");
        }
    }

    /// Move selection up in search results
    pub fn search_select_up(&mut self) {
        if !self.search_state.results.is_empty() {
            self.search_state.selected_index = self.search_state.selected_index.saturating_sub(1);
        }
    }

    /// Move selection down in search results
    pub fn search_select_down(&mut self) {
        if !self.search_state.results.is_empty() {
            self.search_state.selected_index = (self.search_state.selected_index + 1)
                .min(self.search_state.results.len().saturating_sub(1));
        }
    }

    /// Toggle the document modal
    pub fn toggle_search_modal(&mut self) {
        if !self.search_state.results.is_empty() {
            self.search_state.showing_modal = !self.search_state.showing_modal;
        }
    }

    /// Close the document modal
    pub fn close_search_modal(&mut self) {
        self.search_state.showing_modal = false;
    }

    /// Toggle the help modal
    pub fn toggle_search_help(&mut self) {
        self.search_state.showing_help = !self.search_state.showing_help;
    }

    /// Close the help modal
    pub fn close_search_help(&mut self) {
        self.search_state.showing_help = false;
    }

    /// Toggle focus between input and results
    pub fn toggle_search_focus(&mut self) {
        self.search_state.focus = match self.search_state.focus {
            SearchFocus::Input => SearchFocus::Results,
            SearchFocus::Results => SearchFocus::Input,
        };
    }

    /// Handle character input for search query
    pub fn search_input_char(&mut self, c: char) {
        self.search_state.query.push(c);
    }

    /// Handle backspace for search query
    pub fn search_input_backspace(&mut self) {
        self.search_state.query.pop();
    }

    /// Clear the search query
    #[allow(dead_code)]
    pub fn search_clear_query(&mut self) {
        self.search_state.query.clear();
    }

    /// Toggle the search mode between semantic and source
    pub fn toggle_search_mode(&mut self) {
        self.search_state.mode = self.search_state.mode.toggle();
        self.status = format!("Search mode: {}", self.search_state.mode.as_str());
    }

    /// Get the currently selected search result
    pub fn get_selected_search_result(&self) -> Option<&SearchResult> {
        self.search_state
            .results
            .get(self.search_state.selected_index)
    }

    // ==================== Manual Entry Modal Methods ====================

    /// Show the manual entry modal (F3)
    pub fn show_manual_entry_modal(&mut self) {
        self.search_state.showing_manual_entry = true;
        self.search_state.manual_entry_text.clear();
        self.search_state.manual_entry_source = "User Context".to_string();
        self.search_state.manual_entry_focus = ManualEntryFocus::Text;
        self.search_state.manual_entry_loading = false;
        self.search_state.manual_entry_text_cursor = 0;
        self.search_state.manual_entry_source_cursor =
            self.search_state.manual_entry_source.chars().count();
        self.search_state.manual_entry_scroll_offset.set(0);
    }

    /// Close the manual entry modal
    pub fn close_manual_entry_modal(&mut self) {
        self.search_state.showing_manual_entry = false;
        self.search_state.manual_entry_text.clear();
        self.search_state.manual_entry_source.clear();
        self.search_state.manual_entry_loading = false;
        self.search_state.manual_entry_text_cursor = 0;
        self.search_state.manual_entry_source_cursor = 0;
    }

    /// Toggle focus between text and source fields in manual entry modal
    pub fn toggle_manual_entry_focus(&mut self) {
        self.search_state.manual_entry_focus = match self.search_state.manual_entry_focus {
            ManualEntryFocus::Text => ManualEntryFocus::Source,
            ManualEntryFocus::Source => ManualEntryFocus::Text,
        };
    }

    /// Handle character input in manual entry modal
    pub fn manual_entry_input_char(&mut self, c: char) {
        match self.search_state.manual_entry_focus {
            ManualEntryFocus::Text => {
                let cursor = self.search_state.manual_entry_text_cursor;
                let byte_pos = self
                    .search_state
                    .manual_entry_text
                    .char_indices()
                    .nth(cursor)
                    .map(|(i, _)| i)
                    .unwrap_or(self.search_state.manual_entry_text.len());
                self.search_state.manual_entry_text.insert(byte_pos, c);
                self.search_state.manual_entry_text_cursor += 1;
            }
            ManualEntryFocus::Source => {
                let cursor = self.search_state.manual_entry_source_cursor;
                let byte_pos = self
                    .search_state
                    .manual_entry_source
                    .char_indices()
                    .nth(cursor)
                    .map(|(i, _)| i)
                    .unwrap_or(self.search_state.manual_entry_source.len());
                self.search_state.manual_entry_source.insert(byte_pos, c);
                self.search_state.manual_entry_source_cursor += 1;
            }
        }
    }

    /// Handle backspace in manual entry modal
    pub fn manual_entry_input_backspace(&mut self) {
        match self.search_state.manual_entry_focus {
            ManualEntryFocus::Text => {
                let cursor = self.search_state.manual_entry_text_cursor;
                if cursor > 0 {
                    let byte_pos = self
                        .search_state
                        .manual_entry_text
                        .char_indices()
                        .nth(cursor - 1)
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    self.search_state.manual_entry_text.remove(byte_pos);
                    self.search_state.manual_entry_text_cursor -= 1;
                }
            }
            ManualEntryFocus::Source => {
                let cursor = self.search_state.manual_entry_source_cursor;
                if cursor > 0 {
                    let byte_pos = self
                        .search_state
                        .manual_entry_source
                        .char_indices()
                        .nth(cursor - 1)
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    self.search_state.manual_entry_source.remove(byte_pos);
                    self.search_state.manual_entry_source_cursor -= 1;
                }
            }
        }
    }

    /// Handle delete key in manual entry modal
    pub fn manual_entry_input_delete(&mut self) {
        match self.search_state.manual_entry_focus {
            ManualEntryFocus::Text => {
                let cursor = self.search_state.manual_entry_text_cursor;
                let char_count = self.search_state.manual_entry_text.chars().count();
                if cursor < char_count {
                    let byte_pos = self
                        .search_state
                        .manual_entry_text
                        .char_indices()
                        .nth(cursor)
                        .map(|(i, _)| i)
                        .unwrap_or(self.search_state.manual_entry_text.len());
                    self.search_state.manual_entry_text.remove(byte_pos);
                }
            }
            ManualEntryFocus::Source => {
                let cursor = self.search_state.manual_entry_source_cursor;
                let char_count = self.search_state.manual_entry_source.chars().count();
                if cursor < char_count {
                    let byte_pos = self
                        .search_state
                        .manual_entry_source
                        .char_indices()
                        .nth(cursor)
                        .map(|(i, _)| i)
                        .unwrap_or(self.search_state.manual_entry_source.len());
                    self.search_state.manual_entry_source.remove(byte_pos);
                }
            }
        }
    }

    /// Handle newline in manual entry modal (only for text field)
    pub fn manual_entry_input_newline(&mut self) {
        if self.search_state.manual_entry_focus == ManualEntryFocus::Text {
            let cursor = self.search_state.manual_entry_text_cursor;
            let byte_pos = self
                .search_state
                .manual_entry_text
                .char_indices()
                .nth(cursor)
                .map(|(i, _)| i)
                .unwrap_or(self.search_state.manual_entry_text.len());
            self.search_state.manual_entry_text.insert(byte_pos, '\n');
            self.search_state.manual_entry_text_cursor += 1;
        }
    }

    /// Move cursor left in manual entry modal
    pub fn manual_entry_cursor_left(&mut self) {
        match self.search_state.manual_entry_focus {
            ManualEntryFocus::Text => {
                if self.search_state.manual_entry_text_cursor > 0 {
                    self.search_state.manual_entry_text_cursor -= 1;
                }
            }
            ManualEntryFocus::Source => {
                if self.search_state.manual_entry_source_cursor > 0 {
                    self.search_state.manual_entry_source_cursor -= 1;
                }
            }
        }
    }

    /// Move cursor right in manual entry modal
    pub fn manual_entry_cursor_right(&mut self) {
        match self.search_state.manual_entry_focus {
            ManualEntryFocus::Text => {
                let max = self.search_state.manual_entry_text.chars().count();
                if self.search_state.manual_entry_text_cursor < max {
                    self.search_state.manual_entry_text_cursor += 1;
                }
            }
            ManualEntryFocus::Source => {
                let max = self.search_state.manual_entry_source.chars().count();
                if self.search_state.manual_entry_source_cursor < max {
                    self.search_state.manual_entry_source_cursor += 1;
                }
            }
        }
    }

    /// Move cursor up in manual entry modal (text field only)
    pub fn manual_entry_cursor_up(&mut self) {
        if self.search_state.manual_entry_focus != ManualEntryFocus::Text {
            return;
        }

        let text = &self.search_state.manual_entry_text;
        let cursor = self.search_state.manual_entry_text_cursor;

        // Find current line and column
        let mut current_line = 0;
        let mut current_col = 0;
        let mut line_start = 0;

        for (i, c) in text.chars().enumerate() {
            if i == cursor {
                break;
            }
            if c == '\n' {
                current_line += 1;
                current_col = 0;
                line_start = i + 1;
            } else {
                current_col += 1;
            }
        }

        // If on first line, can't go up
        if current_line == 0 {
            return;
        }

        // Find previous line start and length
        let mut prev_line_start = 0;
        let mut prev_line_len = 0;
        let mut line = 0;

        for (i, c) in text.chars().enumerate() {
            if line == current_line - 1 {
                if c == '\n' {
                    prev_line_len = i - prev_line_start;
                    break;
                }
            } else if c == '\n' {
                line += 1;
                if line == current_line - 1 {
                    prev_line_start = i + 1;
                }
            }
        }

        // Handle case where previous line is the last scanned
        if line == current_line - 1 && prev_line_len == 0 {
            prev_line_len = line_start - prev_line_start - 1;
        }

        // Move to same column on previous line, or end if line is shorter
        let new_col = current_col.min(prev_line_len);
        self.search_state.manual_entry_text_cursor = prev_line_start + new_col;
    }

    /// Move cursor down in manual entry modal (text field only)
    pub fn manual_entry_cursor_down(&mut self) {
        if self.search_state.manual_entry_focus != ManualEntryFocus::Text {
            return;
        }

        let text = &self.search_state.manual_entry_text;
        let cursor = self.search_state.manual_entry_text_cursor;
        let char_count = text.chars().count();

        // Find current column
        let mut current_col = 0;
        let mut found_newline_before_cursor = false;

        for (i, c) in text.chars().enumerate() {
            if i == cursor {
                break;
            }
            if c == '\n' {
                current_col = 0;
                found_newline_before_cursor = true;
            } else {
                current_col += 1;
            }
        }

        // If no newline in text, or cursor is before first newline, start col calculation from 0
        if !found_newline_before_cursor {
            current_col = cursor;
        }

        // Find next line start
        let mut next_line_start = None;

        for (i, c) in text.chars().enumerate().skip(cursor) {
            if c == '\n' {
                next_line_start = Some(i + 1);
                break;
            }
        }

        // If no next line, stay put
        let Some(next_start) = next_line_start else {
            return;
        };

        // Find next line length
        let mut next_line_len = 0;
        for (i, c) in text.chars().enumerate().skip(next_start) {
            if c == '\n' {
                next_line_len = i - next_start;
                break;
            }
            next_line_len = i - next_start + 1;
        }

        // Move to same column on next line, or end if line is shorter
        let new_col = current_col.min(next_line_len);
        let new_cursor = next_start + new_col;
        self.search_state.manual_entry_text_cursor = new_cursor.min(char_count);
    }

    /// Move cursor to start of line in manual entry modal
    pub fn manual_entry_cursor_home(&mut self) {
        match self.search_state.manual_entry_focus {
            ManualEntryFocus::Text => {
                let text = &self.search_state.manual_entry_text;
                let cursor = self.search_state.manual_entry_text_cursor;

                // Find start of current line
                let mut line_start = 0;
                for (i, c) in text.chars().enumerate() {
                    if i == cursor {
                        break;
                    }
                    if c == '\n' {
                        line_start = i + 1;
                    }
                }
                self.search_state.manual_entry_text_cursor = line_start;
            }
            ManualEntryFocus::Source => {
                self.search_state.manual_entry_source_cursor = 0;
            }
        }
    }

    /// Move cursor to end of line in manual entry modal
    pub fn manual_entry_cursor_end(&mut self) {
        match self.search_state.manual_entry_focus {
            ManualEntryFocus::Text => {
                let text = &self.search_state.manual_entry_text;
                let cursor = self.search_state.manual_entry_text_cursor;
                let char_count = text.chars().count();

                // Find end of current line
                let mut line_end = char_count;
                for (i, c) in text.chars().enumerate().skip(cursor) {
                    if c == '\n' {
                        line_end = i;
                        break;
                    }
                }
                self.search_state.manual_entry_text_cursor = line_end;
            }
            ManualEntryFocus::Source => {
                self.search_state.manual_entry_source_cursor =
                    self.search_state.manual_entry_source.chars().count();
            }
        }
    }

    /// Copy text content to clipboard
    pub fn manual_entry_copy_to_clipboard(&mut self) {
        let text = &self.search_state.manual_entry_text;
        if text.is_empty() {
            self.status = String::from("Nothing to copy");
            return;
        }

        match arboard::Clipboard::new() {
            Ok(mut clipboard) => match clipboard.set_text(text.clone()) {
                Ok(()) => {
                    self.status = format!("Copied {} chars to clipboard", text.chars().count());
                }
                Err(e) => {
                    self.status = format!("Clipboard error: {}", e);
                }
            },
            Err(e) => {
                self.status = format!("Clipboard unavailable: {}", e);
            }
        }
    }

    /// Paste from clipboard into current field
    pub fn manual_entry_paste_from_clipboard(&mut self) {
        let clipboard_text = match arboard::Clipboard::new() {
            Ok(mut clipboard) => match clipboard.get_text() {
                Ok(text) => text,
                Err(e) => {
                    self.status = format!("Clipboard read error: {}", e);
                    return;
                }
            },
            Err(e) => {
                self.status = format!("Clipboard unavailable: {}", e);
                return;
            }
        };

        if clipboard_text.is_empty() {
            self.status = String::from("Clipboard is empty");
            return;
        }

        match self.search_state.manual_entry_focus {
            ManualEntryFocus::Text => {
                // Insert clipboard text at cursor position
                let cursor = self.search_state.manual_entry_text_cursor;
                let byte_pos = self
                    .search_state
                    .manual_entry_text
                    .char_indices()
                    .nth(cursor)
                    .map(|(i, _)| i)
                    .unwrap_or(self.search_state.manual_entry_text.len());
                self.search_state
                    .manual_entry_text
                    .insert_str(byte_pos, &clipboard_text);
                self.search_state.manual_entry_text_cursor += clipboard_text.chars().count();
                self.status = format!("Pasted {} chars", clipboard_text.chars().count());
            }
            ManualEntryFocus::Source => {
                // For source field, only take first line (no newlines allowed)
                let first_line = clipboard_text.lines().next().unwrap_or("");
                let cursor = self.search_state.manual_entry_source_cursor;
                let byte_pos = self
                    .search_state
                    .manual_entry_source
                    .char_indices()
                    .nth(cursor)
                    .map(|(i, _)| i)
                    .unwrap_or(self.search_state.manual_entry_source.len());
                self.search_state
                    .manual_entry_source
                    .insert_str(byte_pos, first_line);
                self.search_state.manual_entry_source_cursor += first_line.chars().count();
                self.status = format!("Pasted {} chars", first_line.chars().count());
            }
        }
    }

    /// Submit the manual entry to ChromaDB via search tool
    pub async fn submit_manual_entry(&mut self) {
        let text = self.search_state.manual_entry_text.trim().to_string();
        if text.is_empty() {
            self.status = String::from("Cannot add empty text");
            return;
        }

        let source = if self.search_state.manual_entry_source.trim().is_empty() {
            "User Context".to_string()
        } else {
            self.search_state.manual_entry_source.trim().to_string()
        };

        self.search_state.manual_entry_loading = true;
        self.status = String::from("Adding to ChromaDB...");

        let url = format!("{}/add", self.search_tool_url);
        let client = reqwest::Client::new();

        match client
            .post(&url)
            .json(&serde_json::json!({
                "text": text,
                "source": source
            }))
            .timeout(std::time::Duration::from_secs(30))
            .send()
            .await
        {
            Ok(response) => {
                if response.status().is_success() {
                    match response.json::<serde_json::Value>().await {
                        Ok(data) => {
                            if data
                                .get("success")
                                .and_then(|v| v.as_bool())
                                .unwrap_or(false)
                            {
                                let doc_id = data
                                    .get("result")
                                    .and_then(|r| r.get("id"))
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("unknown")
                                    .to_string();

                                // Calculate tokens for this document
                                let doc_tokens = tiktoken_rs::cl100k_base()
                                    .map(|bpe| bpe.encode_with_special_tokens(&text).len())
                                    .unwrap_or(0);

                                // Add to context_documents SQLite table
                                if let Ok(conn) = rusqlite::Connection::open(&self.context_db_path)
                                {
                                    let _ = conn.execute(
                                        "CREATE TABLE IF NOT EXISTS context_documents (
                                            id TEXT PRIMARY KEY,
                                            text TEXT NOT NULL,
                                            source TEXT,
                                            added_at TEXT DEFAULT CURRENT_TIMESTAMP
                                        )",
                                        [],
                                    );

                                    let _ = conn.execute(
                                        "INSERT OR IGNORE INTO context_documents (id, text, source) VALUES (?1, ?2, ?3)",
                                        [&doc_id, &text, &source],
                                    );
                                }

                                // Create a SearchResult and add to results list
                                let new_result = SearchResult {
                                    id: doc_id.clone(),
                                    text: text.clone(),
                                    source: Some(source.clone()),
                                    distance: None,
                                    is_in_context: true,
                                };

                                // Insert at the beginning of results so it's visible
                                self.search_state.results.insert(0, new_result);

                                // Update context tracking
                                self.search_state.context_ids.insert(doc_id.clone());
                                self.search_state.context_token_count += doc_tokens;

                                // Select the new entry
                                self.search_state.selected_index = 0;
                                self.search_state.scroll_offset = 0;

                                self.status = format!("Added to ChromaDB and context: {}", doc_id);
                                self.close_manual_entry_modal();
                            } else {
                                let error = data
                                    .get("error")
                                    .and_then(|e| e.get("message"))
                                    .and_then(|m| m.as_str())
                                    .unwrap_or("Unknown error");
                                self.status = format!("Failed: {}", error);
                            }
                        }
                        Err(e) => {
                            self.status = format!("JSON error: {}", e);
                        }
                    }
                } else {
                    self.status = format!("HTTP error: {}", response.status());
                }
            }
            Err(e) => {
                self.status = format!("Request error: {}", e);
            }
        }

        self.search_state.manual_entry_loading = false;
    }

    /// Show the quit confirmation modal
    pub fn show_quit_modal(&mut self) {
        self.showing_quit_modal = true;
    }

    /// Hide the quit confirmation modal
    pub fn hide_quit_modal(&mut self) {
        self.showing_quit_modal = false;
    }

    /// Confirm quit
    pub fn confirm_quit(&mut self) {
        self.should_quit = true;
    }
}
