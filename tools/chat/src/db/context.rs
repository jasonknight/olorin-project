//! Context database monitoring

#![allow(dead_code)]

use anyhow::Result;
use chrono::Local;
use rusqlite::Connection;
use std::collections::HashSet;
use std::path::PathBuf;

use crate::message::ContextInfo;

/// Context database reader
pub struct ContextDb {
    path: PathBuf,
    seen_prompt_ids: HashSet<String>,
    last_seen_timestamp: Option<String>,
}

impl ContextDb {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            seen_prompt_ids: HashSet::new(),
            last_seen_timestamp: None,
        }
    }

    /// Check if the database file exists
    pub fn exists(&self) -> bool {
        self.path.exists()
    }

    /// Get new context retrievals since last check
    pub fn get_new_contexts(&mut self) -> Result<Vec<ContextInfo>> {
        if !self.exists() {
            return Ok(Vec::new());
        }

        let conn = Connection::open(&self.path)?;

        // Get distinct prompt_ids with their context info
        let query = if let Some(ref last_ts) = self.last_seen_timestamp {
            format!(
                r#"
                SELECT
                    prompt_id,
                    COUNT(*) as chunk_count,
                    GROUP_CONCAT(DISTINCT source) as sources,
                    AVG(distance) as avg_distance,
                    MAX(added_at) as last_added
                FROM contexts
                WHERE added_at > '{}'
                GROUP BY prompt_id
                ORDER BY last_added ASC
                "#,
                last_ts
            )
        } else {
            r#"
            SELECT
                prompt_id,
                COUNT(*) as chunk_count,
                GROUP_CONCAT(DISTINCT source) as sources,
                AVG(distance) as avg_distance,
                MAX(added_at) as last_added
            FROM contexts
            GROUP BY prompt_id
            ORDER BY last_added ASC
            "#
            .to_string()
        };

        let mut stmt = conn.prepare(&query)?;

        let contexts: Vec<ContextInfo> = stmt
            .query_map([], |row| {
                let prompt_id: String = row.get(0)?;
                let chunk_count: i64 = row.get(1)?;
                let sources_str: Option<String> = row.get(2)?;
                let avg_distance: f64 = row.get(3)?;
                let _last_added: String = row.get(4)?;

                let sources: Vec<String> = sources_str
                    .map(|s| s.split(',').map(String::from).collect())
                    .unwrap_or_default();

                Ok(ContextInfo {
                    prompt_id,
                    chunk_count: chunk_count as usize,
                    sources,
                    avg_distance,
                    retrieved_at: Local::now(), // We'll use current time for display
                })
            })?
            .filter_map(|r| r.ok())
            .filter(|ctx| !self.seen_prompt_ids.contains(&ctx.prompt_id))
            .collect();

        // Update tracking
        for ctx in &contexts {
            self.seen_prompt_ids.insert(ctx.prompt_id.clone());
        }

        // Update last seen timestamp
        if !contexts.is_empty() {
            self.last_seen_timestamp =
                Some(Local::now().format("%Y-%m-%d %H:%M:%S%.6f").to_string());
        }

        Ok(contexts)
    }

    /// Get statistics about the context database
    pub fn get_stats(&self) -> Result<ContextStats> {
        if !self.exists() {
            return Ok(ContextStats::default());
        }

        let conn = Connection::open(&self.path)?;

        let total_contexts: i64 = conn
            .query_row("SELECT COUNT(*) FROM contexts", [], |row| row.get(0))
            .unwrap_or(0);

        let unique_prompts: i64 = conn
            .query_row(
                "SELECT COUNT(DISTINCT prompt_id) FROM contexts",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);

        Ok(ContextStats {
            total_contexts: total_contexts as usize,
            unique_prompts: unique_prompts as usize,
        })
    }
}

#[derive(Debug, Default)]
pub struct ContextStats {
    pub total_contexts: usize,
    pub unique_prompts: usize,
}
