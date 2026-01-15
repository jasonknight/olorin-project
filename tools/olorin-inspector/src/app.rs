//! Application state management for olorin-inspector

use anyhow::Result;

use crate::db::{
    ChromaDbSource, DatabaseInfo, DatabaseSource, DatabaseType, DbError, Record, SqliteChat,
    SqliteContext, SqliteFileTracker, SqliteState,
};

/// Configuration for a failed ChromaDB connection (for retry)
#[derive(Debug, Clone)]
pub struct FailedChromaDbConfig {
    pub host: String,
    pub port: u16,
    pub collection: String,
}

/// How often to check ChromaDB health (in ticks)
const HEALTH_CHECK_INTERVAL: u64 = 30; // ~7.5 seconds at 250ms tick rate

/// How often to refresh record counts (in ticks)
const COUNT_REFRESH_INTERVAL: u64 = 20; // ~5 seconds at 250ms tick rate

/// Maximum number of records to keep in memory
const MAX_RECORDS_IN_MEMORY: usize = 500;

/// Number of records to fetch per page
const RECORDS_PER_PAGE: usize = 50;

/// Number of records to fetch when scrolling
const SCROLL_FETCH_SIZE: usize = 25;

/// Currently focused panel
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FocusedPanel {
    DatabaseList,
    RecordsTable,
    QueryInput,
}

/// Modal dialog state for clear database confirmation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClearDbModal {
    /// Which button is selected (true = Yes, false = No)
    pub yes_selected: bool,
}

/// Modal dialog state for viewing record details
#[derive(Debug, Clone)]
pub struct RecordDetailModal {
    /// The record being displayed
    pub record: Record,
    /// Current scroll position within the modal
    pub scroll: usize,
}

impl FocusedPanel {
    pub fn next(self) -> Self {
        match self {
            FocusedPanel::DatabaseList => FocusedPanel::RecordsTable,
            FocusedPanel::RecordsTable => FocusedPanel::QueryInput,
            FocusedPanel::QueryInput => FocusedPanel::DatabaseList,
        }
    }
}

/// Main application state
pub struct App {
    /// Currently focused panel
    pub focus: FocusedPanel,

    /// List of all database sources
    pub databases: Vec<Box<dyn DatabaseSource>>,

    /// Currently selected database index
    pub selected_db: usize,

    /// Cached records for the selected database
    pub records: Vec<Record>,

    /// Current scroll offset in records table
    pub record_scroll: usize,

    /// Query input buffer
    pub query_input: String,

    /// Cursor position in query input
    pub cursor_position: usize,

    /// Whether the app should exit
    pub should_quit: bool,

    /// Status message to display
    pub status_message: Option<String>,

    /// Clear database confirmation modal (None = hidden)
    pub clear_modal: Option<ClearDbModal>,

    /// Record detail modal (None = hidden)
    pub detail_modal: Option<RecordDetailModal>,

    /// Configuration
    pub config: olorin_config::Config,

    /// Error messages from database loading
    pub load_errors: Vec<String>,

    /// Failed ChromaDB configuration (for retry)
    pub failed_chromadb: Option<FailedChromaDbConfig>,

    /// Tick counter for periodic health checks
    tick_count: u64,
}

impl App {
    /// Create a new application instance
    pub fn new() -> Result<Self> {
        let config = olorin_config::Config::new(None, true)?;

        let mut app = Self {
            focus: FocusedPanel::DatabaseList,
            databases: Vec::new(),
            selected_db: 0,
            records: Vec::new(),
            record_scroll: 0,
            query_input: String::new(),
            cursor_position: 0,
            should_quit: false,
            status_message: Some("Loading databases...".to_string()),
            clear_modal: None,
            detail_modal: None,
            config,
            load_errors: Vec::new(),
            failed_chromadb: None,
            tick_count: 0,
        };

        app.load_databases();

        if !app.databases.is_empty() {
            app.refresh_records();
            let mut msg = format!("Loaded {} databases", app.databases.len());
            if app.failed_chromadb.is_some() {
                msg.push_str(" (ChromaDB offline - press R to retry)");
            }
            app.status_message = Some(msg);
        } else if !app.load_errors.is_empty() {
            app.status_message = Some(format!("Errors: {}", app.load_errors.join("; ")));
        } else {
            app.status_message = Some("No databases found".to_string());
        }

        Ok(app)
    }

    /// Load all available databases
    fn load_databases(&mut self) {
        // SQLite file trackers - load paths from config
        let sqlite_trackers = [
            (
                "Markdown Tracker",
                "TRACKING_DB",
                "./hippocampus/data/tracking.db",
            ),
            (
                "PDF Tracker",
                "PDF_TRACKING_DB",
                "./hippocampus/data/pdf_tracking.db",
            ),
            (
                "Ebook Tracker",
                "EBOOK_TRACKING_DB",
                "./hippocampus/data/ebook_tracking.db",
            ),
            (
                "TXT Tracker",
                "TXT_TRACKING_DB",
                "./hippocampus/data/txt_tracking.db",
            ),
            (
                "Office Tracker",
                "OFFICE_TRACKING_DB",
                "./hippocampus/data/office_tracking.db",
            ),
        ];

        for (name, config_key, default_path) in sqlite_trackers {
            if let Some(path) = self.config.get_path(config_key, Some(default_path)) {
                match SqliteFileTracker::new(name, &path) {
                    Ok(db) => self.databases.push(Box::new(db)),
                    Err(DbError::NotFound(_)) => {
                        // Silent skip for non-existent databases
                    }
                    Err(e) => {
                        self.load_errors.push(format!("{}: {}", name, e));
                    }
                }
            }
        }

        // Context store
        if let Some(context_path) = self.config.get_path(
            "HIPPOCAMPUS_CONTEXT_DB",
            Some("./hippocampus/data/context.db"),
        ) {
            match SqliteContext::new("Context Store", &context_path) {
                Ok(db) => self.databases.push(Box::new(db)),
                Err(DbError::NotFound(_)) => {}
                Err(e) => {
                    self.load_errors.push(format!("Context Store: {}", e));
                }
            }
        }

        // Chat history
        if let Some(chat_path) = self
            .config
            .get_path("CHAT_DB_PATH", Some("./cortex/data/chat.db"))
        {
            match SqliteChat::new("Chat History", &chat_path) {
                Ok(db) => self.databases.push(Box::new(db)),
                Err(DbError::NotFound(_)) => {}
                Err(e) => {
                    self.load_errors.push(format!("Chat History: {}", e));
                }
            }
        }

        // System state database
        if let Some(state_path) = self
            .config
            .get_path("STATE_DB_PATH", Some("./data/state.db"))
        {
            match SqliteState::new("System State", &state_path) {
                Ok(db) => self.databases.push(Box::new(db)),
                Err(DbError::NotFound(_)) => {}
                Err(e) => {
                    self.load_errors.push(format!("System State: {}", e));
                }
            }
        }

        // ChromaDB
        let chromadb_host = self
            .config
            .get("CHROMADB_HOST", Some("localhost"))
            .unwrap_or_else(|| "localhost".to_string());
        let chromadb_port = self
            .config
            .get_int("CHROMADB_PORT", Some(8000))
            .unwrap_or(8000) as u16;
        let collection = self
            .config
            .get("CHROMADB_COLLECTION", Some("documents"))
            .unwrap_or_else(|| "documents".to_string());

        match ChromaDbSource::new(&chromadb_host, chromadb_port, &collection) {
            Ok(db) => {
                self.databases.push(Box::new(db));
                self.failed_chromadb = None; // Clear any previous failure
            }
            Err(DbError::Connection(e)) => {
                // Store config for retry
                self.failed_chromadb = Some(FailedChromaDbConfig {
                    host: chromadb_host,
                    port: chromadb_port,
                    collection,
                });
                self.load_errors
                    .push(format!("ChromaDB: {} (press R to retry)", e));
            }
            Err(e) => {
                self.load_errors.push(format!("ChromaDB: {}", e));
            }
        }
    }

    /// Get the currently selected database info
    pub fn current_db_info(&self) -> Option<&DatabaseInfo> {
        self.databases.get(self.selected_db).map(|db| db.info())
    }

    /// Cycle to the next focus panel
    pub fn next_focus(&mut self) {
        self.focus = self.focus.next();
    }

    /// Select the next database in the list
    pub fn select_next_db(&mut self) {
        if self.selected_db < self.databases.len().saturating_sub(1) {
            self.selected_db += 1;
            self.refresh_records();
        }
    }

    /// Select the previous database in the list
    pub fn select_prev_db(&mut self) {
        if self.selected_db > 0 {
            self.selected_db -= 1;
            self.refresh_records();
        }
    }

    /// Scroll records down
    pub fn scroll_records_down(&mut self) {
        let max_scroll = self.records.len().saturating_sub(1);
        if self.record_scroll < max_scroll {
            self.record_scroll += 1;

            // Load more records if near bottom (infinite scroll)
            if self.record_scroll > self.records.len().saturating_sub(10) {
                self.load_more_records();
            }
        }
    }

    /// Scroll records up
    pub fn scroll_records_up(&mut self) {
        self.record_scroll = self.record_scroll.saturating_sub(1);
    }

    /// Page down in records
    pub fn page_down(&mut self) {
        for _ in 0..10 {
            self.scroll_records_down();
        }
    }

    /// Page up in records
    pub fn page_up(&mut self) {
        for _ in 0..10 {
            self.scroll_records_up();
        }
    }

    /// Refresh records from the current database
    pub fn refresh_records(&mut self) {
        if let Some(db) = self.databases.get(self.selected_db) {
            match db.fetch_recent(RECORDS_PER_PAGE) {
                Ok(records) => {
                    self.records = records;
                    self.record_scroll = 0;
                    self.status_message = Some(format!(
                        "Loaded {} records from {}",
                        self.records.len(),
                        db.info().name
                    ));
                }
                Err(e) => {
                    self.status_message = Some(format!("Error: {}", e));
                }
            }
        }
    }

    /// Load more records for infinite scroll
    pub fn load_more_records(&mut self) {
        if self.records.len() >= MAX_RECORDS_IN_MEMORY {
            return;
        }

        if let Some(db) = self.databases.get(self.selected_db) {
            if let Some(last_record) = self.records.last() {
                if let Some(timestamp) = &last_record.timestamp {
                    match db.fetch_before(timestamp, SCROLL_FETCH_SIZE) {
                        Ok(more_records) => {
                            if !more_records.is_empty() {
                                self.records.extend(more_records);
                            }
                        }
                        Err(_) => {
                            // Silently ignore fetch errors during scroll
                        }
                    }
                }
            }
        }
    }

    /// Execute the current query
    pub fn execute_query(&mut self) {
        if self.query_input.trim().is_empty() {
            self.status_message = Some("Enter a query first".to_string());
            return;
        }

        if let Some(db) = self.databases.get(self.selected_db) {
            match db.execute_query(&self.query_input) {
                Ok(records) => {
                    let count = records.len();
                    self.records = records;
                    self.record_scroll = 0;
                    self.status_message = Some(format!("Query returned {} results", count));
                }
                Err(e) => {
                    self.status_message = Some(format!("Query error: {}", e));
                }
            }
        }
    }

    /// Handle character input for query
    pub fn input_char(&mut self, c: char) {
        self.query_input.insert(self.cursor_position, c);
        self.cursor_position += 1;
    }

    /// Handle backspace in query input
    pub fn input_backspace(&mut self) {
        if self.cursor_position > 0 {
            self.cursor_position -= 1;
            self.query_input.remove(self.cursor_position);
        }
    }

    /// Handle delete in query input
    pub fn input_delete(&mut self) {
        if self.cursor_position < self.query_input.len() {
            self.query_input.remove(self.cursor_position);
        }
    }

    /// Move cursor left in query input
    pub fn cursor_left(&mut self) {
        self.cursor_position = self.cursor_position.saturating_sub(1);
    }

    /// Move cursor right in query input
    pub fn cursor_right(&mut self) {
        if self.cursor_position < self.query_input.len() {
            self.cursor_position += 1;
        }
    }

    /// Move cursor to start of line
    pub fn cursor_home(&mut self) {
        self.cursor_position = 0;
    }

    /// Move cursor to end of line
    pub fn cursor_end(&mut self) {
        self.cursor_position = self.query_input.len();
    }

    /// Clear the query input
    pub fn clear_query(&mut self) {
        self.query_input.clear();
        self.cursor_position = 0;
    }

    /// Check for config hot-reload
    pub fn check_config_reload(&mut self) {
        if self.config.reload() {
            self.status_message = Some("Configuration reloaded".to_string());
        }
    }

    /// Get visible columns for the current database
    pub fn current_columns(&self) -> Vec<String> {
        self.current_db_info()
            .map(|info| info.columns.clone())
            .unwrap_or_default()
    }

    /// Show the clear database confirmation modal
    pub fn show_clear_modal(&mut self) {
        self.clear_modal = Some(ClearDbModal {
            yes_selected: false,
        });
    }

    /// Hide the clear database confirmation modal
    pub fn hide_clear_modal(&mut self) {
        self.clear_modal = None;
    }

    /// Toggle the selection in the clear modal
    pub fn toggle_clear_modal_selection(&mut self) {
        if let Some(ref mut modal) = self.clear_modal {
            modal.yes_selected = !modal.yes_selected;
        }
    }

    /// Set the clear modal selection directly
    pub fn set_clear_modal_selection(&mut self, yes: bool) {
        if let Some(ref mut modal) = self.clear_modal {
            modal.yes_selected = yes;
        }
    }

    /// Clear the currently selected database
    pub fn clear_current_database(&mut self) {
        if let Some(db) = self.databases.get_mut(self.selected_db) {
            let db_name = db.info().name.clone();
            match db.clear_database() {
                Ok(deleted) => {
                    self.status_message =
                        Some(format!("Cleared {} - deleted {} records", db_name, deleted));
                    self.records.clear();
                    self.record_scroll = 0;
                    // Refresh count
                    let _ = db.refresh_count();
                }
                Err(e) => {
                    self.status_message = Some(format!("Failed to clear {}: {}", db_name, e));
                }
            }
        }
        self.hide_clear_modal();
    }

    /// Handle periodic tick events
    pub fn on_tick(&mut self) {
        self.tick_count += 1;

        // Check for config hot-reload
        self.check_config_reload();

        // Periodic health check for network-based databases
        if self.tick_count % HEALTH_CHECK_INTERVAL == 0 {
            self.check_database_health();
        }

        // Periodic count refresh for all databases
        if self.tick_count % COUNT_REFRESH_INTERVAL == 0 {
            self.refresh_database_counts();
        }
    }

    /// Refresh record counts for all databases
    fn refresh_database_counts(&mut self) {
        for db in &mut self.databases {
            // Only refresh if database is available
            if db.info().connection_state.is_available() {
                let _ = db.refresh_count();
            }
        }
    }

    /// Check health of network-based databases (ChromaDB)
    fn check_database_health(&mut self) {
        for db in &mut self.databases {
            if db.info().db_type == DatabaseType::ChromaDB {
                let was_connected = db.info().connection_state.is_available();
                let is_connected = db.health_check();

                // Notify on state change
                if was_connected && !is_connected {
                    self.status_message = Some(format!("{} went offline", db.info().name));
                } else if !was_connected && is_connected {
                    self.status_message = Some(format!("{} reconnected", db.info().name));
                }
            }
        }
    }

    /// Retry connecting to ChromaDB if it failed at startup
    pub fn retry_chromadb_connection(&mut self) {
        // First, check if we have a failed config to retry
        let config = match self.failed_chromadb.take() {
            Some(c) => c,
            None => {
                // Check if we already have ChromaDB in the list and it's disconnected
                let chromadb_idx = self
                    .databases
                    .iter()
                    .position(|db| db.info().db_type == DatabaseType::ChromaDB);

                if let Some(idx) = chromadb_idx {
                    // Health check the existing ChromaDB
                    if self.databases[idx].health_check() {
                        self.status_message = Some("ChromaDB is already connected".to_string());
                    } else {
                        self.status_message = Some(format!(
                            "ChromaDB still offline: {}",
                            self.databases[idx]
                                .info()
                                .connection_state
                                .error_message()
                                .unwrap_or("unknown error")
                        ));
                    }
                } else {
                    self.status_message = Some("No ChromaDB to retry".to_string());
                }
                return;
            }
        };

        self.status_message = Some("Retrying ChromaDB connection...".to_string());

        match ChromaDbSource::new(&config.host, config.port, &config.collection) {
            Ok(db) => {
                self.databases.push(Box::new(db));
                // Remove the error from load_errors
                self.load_errors.retain(|e| !e.contains("ChromaDB"));
                self.status_message = Some(format!(
                    "ChromaDB connected! ({} databases total)",
                    self.databases.len()
                ));
            }
            Err(DbError::Connection(e)) => {
                // Put the config back for another retry
                self.failed_chromadb = Some(config);
                self.status_message = Some(format!("ChromaDB still offline: {}", e));
            }
            Err(e) => {
                self.status_message = Some(format!("ChromaDB error: {}", e));
            }
        }
    }

    /// Check if there's a failed ChromaDB that can be retried
    #[allow(dead_code)]
    pub fn has_failed_chromadb(&self) -> bool {
        self.failed_chromadb.is_some()
    }

    /// Show the record detail modal for the currently selected record
    pub fn show_detail_modal(&mut self) {
        if let Some(record) = self.records.get(self.record_scroll).cloned() {
            self.detail_modal = Some(RecordDetailModal { record, scroll: 0 });
        }
    }

    /// Hide the record detail modal
    pub fn hide_detail_modal(&mut self) {
        self.detail_modal = None;
    }

    /// Scroll up in the detail modal
    pub fn detail_modal_scroll_up(&mut self) {
        if let Some(ref mut modal) = self.detail_modal {
            modal.scroll = modal.scroll.saturating_sub(1);
        }
    }

    /// Scroll down in the detail modal
    pub fn detail_modal_scroll_down(&mut self) {
        if let Some(ref mut modal) = self.detail_modal {
            modal.scroll += 1;
        }
    }

    /// Page up in the detail modal
    pub fn detail_modal_page_up(&mut self) {
        if let Some(ref mut modal) = self.detail_modal {
            modal.scroll = modal.scroll.saturating_sub(10);
        }
    }

    /// Page down in the detail modal
    pub fn detail_modal_page_down(&mut self) {
        if let Some(ref mut modal) = self.detail_modal {
            modal.scroll += 10;
        }
    }
}
