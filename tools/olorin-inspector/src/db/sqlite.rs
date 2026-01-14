//! SQLite database implementations

use std::path::Path;

use rusqlite::{Connection, OpenFlags};

use super::traits::{DatabaseInfo, DatabaseSource, DatabaseType, DbError, Record};

/// SQLite database source for file tracking databases (processed_files schema)
pub struct SqliteFileTracker {
    info: DatabaseInfo,
    conn: Connection,
}

impl SqliteFileTracker {
    /// Create a new file tracker database source
    ///
    /// # Arguments
    /// * `name` - Display name for this database
    /// * `path` - Path to the SQLite database file
    pub fn new(name: &str, path: &Path) -> Result<Self, DbError> {
        if !path.exists() {
            return Err(DbError::NotFound(path.display().to_string()));
        }

        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;

        let columns = vec![
            "file_path".to_string(),
            "status".to_string(),
            "processed_at".to_string(),
            "chunk_count".to_string(),
            "file_size".to_string(),
            "error_message".to_string(),
        ];

        let count: usize =
            conn.query_row("SELECT COUNT(*) FROM processed_files", [], |row| row.get(0))?;

        Ok(Self {
            info: DatabaseInfo {
                name: name.to_string(),
                db_type: DatabaseType::SqliteFileTracker,
                path: path.display().to_string(),
                record_count: count,
                columns,
                table_name: "processed_files".to_string(),
            },
            conn,
        })
    }

    fn row_to_record(row: &rusqlite::Row) -> rusqlite::Result<Record> {
        let file_path: String = row.get(0)?;
        let content_hash: String = row.get(1)?;
        let file_size: i64 = row.get(2)?;
        let processed_at: String = row.get(3)?;
        let chunk_count: i64 = row.get(4)?;
        let status: String = row.get(5)?;
        let retries: i64 = row.get(6)?;
        let error_message: Option<String> = row.get(7)?;

        let mut record = Record::new();
        record.fields.insert("file_path".to_string(), file_path);
        record.fields.insert("content_hash".to_string(), content_hash);
        record.fields.insert("file_size".to_string(), file_size.to_string());
        record.fields.insert("processed_at".to_string(), processed_at.clone());
        record.fields.insert("chunk_count".to_string(), chunk_count.to_string());
        record.fields.insert("status".to_string(), status);
        record.fields.insert("retries".to_string(), retries.to_string());
        record
            .fields
            .insert("error_message".to_string(), error_message.unwrap_or_default());
        record.timestamp = Some(processed_at);

        Ok(record)
    }
}

impl DatabaseSource for SqliteFileTracker {
    fn info(&self) -> &DatabaseInfo {
        &self.info
    }

    fn fetch_recent(&self, limit: usize) -> Result<Vec<Record>, DbError> {
        let mut stmt = self.conn.prepare(
            "SELECT file_path, content_hash, file_size, processed_at,
                    chunk_count, status, retries, error_message
             FROM processed_files
             ORDER BY processed_at DESC
             LIMIT ?",
        )?;

        let records = stmt.query_map([limit], Self::row_to_record)?;

        records
            .collect::<Result<Vec<_>, _>>()
            .map_err(DbError::from)
    }

    fn fetch_before(&self, before: &str, limit: usize) -> Result<Vec<Record>, DbError> {
        let mut stmt = self.conn.prepare(
            "SELECT file_path, content_hash, file_size, processed_at,
                    chunk_count, status, retries, error_message
             FROM processed_files
             WHERE processed_at < ?
             ORDER BY processed_at DESC
             LIMIT ?",
        )?;

        let records = stmt.query_map((before, limit), Self::row_to_record)?;

        records
            .collect::<Result<Vec<_>, _>>()
            .map_err(DbError::from)
    }

    fn execute_query(&self, query: &str) -> Result<Vec<Record>, DbError> {
        let mut stmt = self.conn.prepare(query).map_err(|e| DbError::Query(e.to_string()))?;

        let column_count = stmt.column_count();
        let column_names: Vec<String> = stmt.column_names().iter().map(|s| s.to_string()).collect();

        let mut records = Vec::new();
        let mut rows = stmt.query([]).map_err(|e| DbError::Query(e.to_string()))?;

        while let Some(row) = rows.next().map_err(|e| DbError::Query(e.to_string()))? {
            let mut record = Record::new();

            for i in 0..column_count {
                let value: String = row
                    .get::<_, rusqlite::types::Value>(i)
                    .map(|v| match v {
                        rusqlite::types::Value::Null => "NULL".to_string(),
                        rusqlite::types::Value::Integer(n) => n.to_string(),
                        rusqlite::types::Value::Real(f) => f.to_string(),
                        rusqlite::types::Value::Text(s) => s,
                        rusqlite::types::Value::Blob(b) => format!("<blob {} bytes>", b.len()),
                    })
                    .unwrap_or_default();

                record.fields.insert(column_names[i].clone(), value);
            }

            // Try to extract timestamp from common column names
            for ts_col in &["processed_at", "created_at", "added_at", "timestamp"] {
                if let Some(ts) = record.fields.get(*ts_col) {
                    record.timestamp = Some(ts.clone());
                    break;
                }
            }

            records.push(record);
        }

        Ok(records)
    }

    fn refresh_count(&mut self) -> Result<usize, DbError> {
        let count: usize =
            self.conn
                .query_row("SELECT COUNT(*) FROM processed_files", [], |row| row.get(0))?;
        self.info.record_count = count;
        Ok(count)
    }

    fn clear_database(&mut self) -> Result<usize, DbError> {
        // Open a write connection to perform the delete
        let write_conn = Connection::open_with_flags(
            &self.info.path,
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;

        let count = self.info.record_count;
        write_conn.execute("DELETE FROM processed_files", [])?;
        self.info.record_count = 0;
        Ok(count)
    }
}

/// SQLite database source for context store (contexts table)
pub struct SqliteContext {
    info: DatabaseInfo,
    conn: Connection,
}

impl SqliteContext {
    pub fn new(name: &str, path: &Path) -> Result<Self, DbError> {
        if !path.exists() {
            return Err(DbError::NotFound(path.display().to_string()));
        }

        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;

        let columns = vec![
            "id".to_string(),
            "prompt_id".to_string(),
            "source".to_string(),
            "h1".to_string(),
            "content".to_string(),
            "added_at".to_string(),
        ];

        let count: usize = conn.query_row("SELECT COUNT(*) FROM contexts", [], |row| row.get(0))?;

        Ok(Self {
            info: DatabaseInfo {
                name: name.to_string(),
                db_type: DatabaseType::SqliteContext,
                path: path.display().to_string(),
                record_count: count,
                columns,
                table_name: "contexts".to_string(),
            },
            conn,
        })
    }

    fn row_to_record(row: &rusqlite::Row) -> rusqlite::Result<Record> {
        let id: String = row.get(0)?;
        let prompt_id: String = row.get(1)?;
        let content: String = row.get(2)?;
        let source: Option<String> = row.get(3)?;
        let h1: Option<String> = row.get(4)?;
        let h2: Option<String> = row.get(5)?;
        let h3: Option<String> = row.get(6)?;
        let chunk_index: Option<i64> = row.get(7)?;
        let distance: Option<f64> = row.get(8)?;
        let added_at: String = row.get(9)?;

        let mut record = Record::new();
        record.fields.insert("id".to_string(), id);
        record.fields.insert("prompt_id".to_string(), prompt_id);
        record.fields.insert(
            "content".to_string(),
            content.chars().take(100).collect::<String>() + "...",
        );
        record
            .fields
            .insert("source".to_string(), source.unwrap_or_default());
        record
            .fields
            .insert("h1".to_string(), h1.unwrap_or_default());
        record
            .fields
            .insert("h2".to_string(), h2.unwrap_or_default());
        record
            .fields
            .insert("h3".to_string(), h3.unwrap_or_default());
        record.fields.insert(
            "chunk_index".to_string(),
            chunk_index.map(|n| n.to_string()).unwrap_or_default(),
        );
        record.fields.insert(
            "distance".to_string(),
            distance.map(|f| format!("{:.4}", f)).unwrap_or_default(),
        );
        record.fields.insert("added_at".to_string(), added_at.clone());
        record.timestamp = Some(added_at);

        Ok(record)
    }
}

impl DatabaseSource for SqliteContext {
    fn info(&self) -> &DatabaseInfo {
        &self.info
    }

    fn fetch_recent(&self, limit: usize) -> Result<Vec<Record>, DbError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, prompt_id, content, source, h1, h2, h3,
                    chunk_index, distance, added_at
             FROM contexts
             ORDER BY added_at DESC
             LIMIT ?",
        )?;

        let records = stmt.query_map([limit], Self::row_to_record)?;

        records
            .collect::<Result<Vec<_>, _>>()
            .map_err(DbError::from)
    }

    fn fetch_before(&self, before: &str, limit: usize) -> Result<Vec<Record>, DbError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, prompt_id, content, source, h1, h2, h3,
                    chunk_index, distance, added_at
             FROM contexts
             WHERE added_at < ?
             ORDER BY added_at DESC
             LIMIT ?",
        )?;

        let records = stmt.query_map((before, limit), Self::row_to_record)?;

        records
            .collect::<Result<Vec<_>, _>>()
            .map_err(DbError::from)
    }

    fn execute_query(&self, query: &str) -> Result<Vec<Record>, DbError> {
        // Reuse the generic query execution from SqliteFileTracker
        let mut stmt = self.conn.prepare(query).map_err(|e| DbError::Query(e.to_string()))?;

        let column_count = stmt.column_count();
        let column_names: Vec<String> = stmt.column_names().iter().map(|s| s.to_string()).collect();

        let mut records = Vec::new();
        let mut rows = stmt.query([]).map_err(|e| DbError::Query(e.to_string()))?;

        while let Some(row) = rows.next().map_err(|e| DbError::Query(e.to_string()))? {
            let mut record = Record::new();

            for i in 0..column_count {
                let value: String = row
                    .get::<_, rusqlite::types::Value>(i)
                    .map(|v| match v {
                        rusqlite::types::Value::Null => "NULL".to_string(),
                        rusqlite::types::Value::Integer(n) => n.to_string(),
                        rusqlite::types::Value::Real(f) => f.to_string(),
                        rusqlite::types::Value::Text(s) => s,
                        rusqlite::types::Value::Blob(b) => format!("<blob {} bytes>", b.len()),
                    })
                    .unwrap_or_default();

                record.fields.insert(column_names[i].clone(), value);
            }

            for ts_col in &["added_at", "created_at", "timestamp"] {
                if let Some(ts) = record.fields.get(*ts_col) {
                    record.timestamp = Some(ts.clone());
                    break;
                }
            }

            records.push(record);
        }

        Ok(records)
    }

    fn refresh_count(&mut self) -> Result<usize, DbError> {
        let count: usize = self
            .conn
            .query_row("SELECT COUNT(*) FROM contexts", [], |row| row.get(0))?;
        self.info.record_count = count;
        Ok(count)
    }

    fn clear_database(&mut self) -> Result<usize, DbError> {
        // Open a write connection to perform the delete
        let write_conn = Connection::open_with_flags(
            &self.info.path,
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;

        let count = self.info.record_count;
        write_conn.execute("DELETE FROM contexts", [])?;
        self.info.record_count = 0;
        Ok(count)
    }
}

/// SQLite database source for chat history (conversations + messages tables)
pub struct SqliteChat {
    info: DatabaseInfo,
    conn: Connection,
}

impl SqliteChat {
    pub fn new(name: &str, path: &Path) -> Result<Self, DbError> {
        if !path.exists() {
            return Err(DbError::NotFound(path.display().to_string()));
        }

        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;

        let columns = vec![
            "id".to_string(),
            "conversation_id".to_string(),
            "role".to_string(),
            "content".to_string(),
            "created_at".to_string(),
        ];

        let count: usize = conn.query_row("SELECT COUNT(*) FROM messages", [], |row| row.get(0))?;

        Ok(Self {
            info: DatabaseInfo {
                name: name.to_string(),
                db_type: DatabaseType::SqliteChat,
                path: path.display().to_string(),
                record_count: count,
                columns,
                table_name: "messages".to_string(),
            },
            conn,
        })
    }

    fn row_to_record(row: &rusqlite::Row) -> rusqlite::Result<Record> {
        let id: String = row.get(0)?;
        let conversation_id: String = row.get(1)?;
        let role: String = row.get(2)?;
        let content: String = row.get(3)?;
        let prompt_id: Option<String> = row.get(4)?;
        let created_at: String = row.get(5)?;

        let mut record = Record::new();
        record.fields.insert("id".to_string(), id);
        record
            .fields
            .insert("conversation_id".to_string(), conversation_id);
        record.fields.insert("role".to_string(), role);
        record.fields.insert(
            "content".to_string(),
            content.chars().take(100).collect::<String>() + "...",
        );
        record
            .fields
            .insert("prompt_id".to_string(), prompt_id.unwrap_or_default());
        record
            .fields
            .insert("created_at".to_string(), created_at.clone());
        record.timestamp = Some(created_at);

        Ok(record)
    }
}

impl DatabaseSource for SqliteChat {
    fn info(&self) -> &DatabaseInfo {
        &self.info
    }

    fn fetch_recent(&self, limit: usize) -> Result<Vec<Record>, DbError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, conversation_id, role, content, prompt_id, created_at
             FROM messages
             ORDER BY created_at DESC
             LIMIT ?",
        )?;

        let records = stmt.query_map([limit], Self::row_to_record)?;

        records
            .collect::<Result<Vec<_>, _>>()
            .map_err(DbError::from)
    }

    fn fetch_before(&self, before: &str, limit: usize) -> Result<Vec<Record>, DbError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, conversation_id, role, content, prompt_id, created_at
             FROM messages
             WHERE created_at < ?
             ORDER BY created_at DESC
             LIMIT ?",
        )?;

        let records = stmt.query_map((before, limit), Self::row_to_record)?;

        records
            .collect::<Result<Vec<_>, _>>()
            .map_err(DbError::from)
    }

    fn execute_query(&self, query: &str) -> Result<Vec<Record>, DbError> {
        let mut stmt = self.conn.prepare(query).map_err(|e| DbError::Query(e.to_string()))?;

        let column_count = stmt.column_count();
        let column_names: Vec<String> = stmt.column_names().iter().map(|s| s.to_string()).collect();

        let mut records = Vec::new();
        let mut rows = stmt.query([]).map_err(|e| DbError::Query(e.to_string()))?;

        while let Some(row) = rows.next().map_err(|e| DbError::Query(e.to_string()))? {
            let mut record = Record::new();

            for i in 0..column_count {
                let value: String = row
                    .get::<_, rusqlite::types::Value>(i)
                    .map(|v| match v {
                        rusqlite::types::Value::Null => "NULL".to_string(),
                        rusqlite::types::Value::Integer(n) => n.to_string(),
                        rusqlite::types::Value::Real(f) => f.to_string(),
                        rusqlite::types::Value::Text(s) => s,
                        rusqlite::types::Value::Blob(b) => format!("<blob {} bytes>", b.len()),
                    })
                    .unwrap_or_default();

                record.fields.insert(column_names[i].clone(), value);
            }

            for ts_col in &["created_at", "last_message_at", "timestamp"] {
                if let Some(ts) = record.fields.get(*ts_col) {
                    record.timestamp = Some(ts.clone());
                    break;
                }
            }

            records.push(record);
        }

        Ok(records)
    }

    fn refresh_count(&mut self) -> Result<usize, DbError> {
        let count: usize = self
            .conn
            .query_row("SELECT COUNT(*) FROM messages", [], |row| row.get(0))?;
        self.info.record_count = count;
        Ok(count)
    }

    fn clear_database(&mut self) -> Result<usize, DbError> {
        // Open a write connection to perform the delete
        let write_conn = Connection::open_with_flags(
            &self.info.path,
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;

        let count = self.info.record_count;
        // Clear both tables (messages depend on conversations via foreign key)
        write_conn.execute("DELETE FROM messages", [])?;
        write_conn.execute("DELETE FROM conversations", [])?;
        self.info.record_count = 0;
        Ok(count)
    }
}
