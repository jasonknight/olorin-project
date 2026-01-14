//! Application state management for olorin-inspector

use std::path::PathBuf;

use anyhow::Result;

use crate::db::{
    ChromaDbSource, DatabaseInfo, DatabaseSource, DbError, Record, SqliteChat, SqliteContext,
    SqliteFileTracker,
};

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

    /// Configuration
    pub config: olorin_config::Config,

    /// Project root path
    pub project_root: PathBuf,

    /// Error messages from database loading
    pub load_errors: Vec<String>,
}

impl App {
    /// Create a new application instance
    pub fn new() -> Result<Self> {
        let config = olorin_config::Config::new(None, true)?;

        // Determine project root from config path
        let project_root = config
            .env_path()
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));

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
            config,
            project_root,
            load_errors: Vec::new(),
        };

        app.load_databases();

        if !app.databases.is_empty() {
            app.refresh_records();
            app.status_message = Some(format!("Loaded {} databases", app.databases.len()));
        } else if !app.load_errors.is_empty() {
            app.status_message = Some(format!("Errors: {}", app.load_errors.join("; ")));
        } else {
            app.status_message = Some("No databases found".to_string());
        }

        Ok(app)
    }

    /// Load all available databases
    fn load_databases(&mut self) {
        // Base paths for hippocampus databases
        let hippocampus_data = self.project_root.join("hippocampus/data");

        // SQLite file trackers
        let sqlite_trackers = [
            ("Markdown Tracker", "tracking.db"),
            ("PDF Tracker", "pdf_tracking.db"),
            ("Ebook Tracker", "ebook_tracking.db"),
            ("TXT Tracker", "txt_tracking.db"),
            ("Office Tracker", "office_tracking.db"),
        ];

        for (name, file) in sqlite_trackers {
            let path = hippocampus_data.join(file);
            match SqliteFileTracker::new(name, &path) {
                Ok(db) => self.databases.push(Box::new(db)),
                Err(DbError::NotFound(_)) => {
                    // Silent skip for non-existent databases
                }
                Err(e) => {
                    self.load_errors
                        .push(format!("{}: {}", name, e));
                }
            }
        }

        // Context store
        let context_path = hippocampus_data.join("context.db");
        match SqliteContext::new("Context Store", &context_path) {
            Ok(db) => self.databases.push(Box::new(db)),
            Err(DbError::NotFound(_)) => {}
            Err(e) => {
                self.load_errors
                    .push(format!("Context Store: {}", e));
            }
        }

        // Chat history
        let chat_path = self.project_root.join("cortex/cortex/data/chat.db");
        match SqliteChat::new("Chat History", &chat_path) {
            Ok(db) => self.databases.push(Box::new(db)),
            Err(DbError::NotFound(_)) => {}
            Err(e) => {
                self.load_errors
                    .push(format!("Chat History: {}", e));
            }
        }

        // ChromaDB
        let chromadb_host = self
            .config
            .get("CHROMADB_HOST", Some("localhost"))
            .unwrap_or_else(|| "localhost".to_string());
        let chromadb_port = self.config.get_int("CHROMADB_PORT", Some(8000)).unwrap_or(8000) as u16;
        let collection = self
            .config
            .get("CHROMADB_COLLECTION", Some("documents"))
            .unwrap_or_else(|| "documents".to_string());

        match ChromaDbSource::new(&chromadb_host, chromadb_port, &collection) {
            Ok(db) => self.databases.push(Box::new(db)),
            Err(DbError::Connection(e)) => {
                self.load_errors
                    .push(format!("ChromaDB: {} (is it running?)", e));
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
        self.clear_modal = Some(ClearDbModal { yes_selected: false });
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
                    self.status_message = Some(format!(
                        "Cleared {} - deleted {} records",
                        db_name, deleted
                    ));
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
}
