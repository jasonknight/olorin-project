//! Database abstraction layer for olorin-inspector
//!
//! Provides a unified interface for querying different database types
//! (SQLite and ChromaDB) used in the Olorin project.

mod chromadb;
mod sqlite;
mod traits;

pub use chromadb::ChromaDbSource;
pub use sqlite::{SqliteChat, SqliteContext, SqliteFileTracker};
pub use traits::{ConnectionState, DatabaseInfo, DatabaseSource, DatabaseType, DbError, Record};
