//! ChromaDB HTTP client implementation

use std::collections::HashMap;

use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

use super::traits::{ConnectionState, DatabaseInfo, DatabaseSource, DatabaseType, DbError, Record};

#[derive(Serialize)]
struct ChromaGetRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    limit: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    offset: Option<usize>,
    include: Vec<String>,
}

#[derive(Serialize)]
struct ChromaDeleteRequest {
    ids: Vec<String>,
}

#[derive(Deserialize, Debug)]
struct ChromaGetResponse {
    ids: Vec<String>,
    #[serde(default)]
    documents: Option<Vec<Option<String>>>,
    #[serde(default)]
    metadatas: Option<Vec<Option<HashMap<String, serde_json::Value>>>>,
}

#[derive(Deserialize, Debug)]
struct ChromaCollectionInfo {
    id: String,
    name: String,
    #[serde(default)]
    #[allow(dead_code)]
    metadata: Option<HashMap<String, serde_json::Value>>,
}

/// ChromaDB database source via HTTP API
pub struct ChromaDbSource {
    info: DatabaseInfo,
    client: Client,
    base_url: String,
    collection_id: String,
    /// Current pagination offset (ChromaDB uses offset-based pagination)
    current_offset: std::cell::Cell<usize>,
}

impl ChromaDbSource {
    /// Create a new ChromaDB source
    ///
    /// # Arguments
    /// * `host` - ChromaDB server host
    /// * `port` - ChromaDB server port
    /// * `collection` - Collection name to inspect
    pub fn new(host: &str, port: u16, collection: &str) -> Result<Self, DbError> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()?;

        // Use v2 API with default tenant and database
        let base_url = format!(
            "http://{}:{}/api/v2/tenants/default_tenant/databases/default_database",
            host, port
        );

        // Get collection by name
        let collections_url = format!("{}/collections", base_url);
        let collections: Vec<ChromaCollectionInfo> = client
            .get(&collections_url)
            .send()
            .map_err(|e| DbError::Connection(format!("Failed to connect to ChromaDB: {}", e)))?
            .json()
            .map_err(|e| DbError::Connection(format!("Invalid response from ChromaDB: {}", e)))?;

        let collection_info = collections
            .iter()
            .find(|c| c.name == collection)
            .ok_or_else(|| DbError::NotFound(format!("Collection '{}' not found", collection)))?;

        // Store the collection UUID for API calls
        let collection_uuid = collection_info.id.clone();

        // Get count using UUID
        let count_url = format!("{}/collections/{}/count", base_url, collection_uuid);
        let count: usize = client
            .get(&count_url)
            .send()
            .map_err(|e| DbError::Connection(e.to_string()))?
            .json()
            .map_err(|e| DbError::Query(format!("Failed to get count: {}", e)))?;

        let columns = vec![
            "id".to_string(),
            "document".to_string(),
            "source".to_string(),
            "h1".to_string(),
            "h2".to_string(),
            "chunk_index".to_string(),
        ];

        Ok(Self {
            info: DatabaseInfo {
                name: format!("ChromaDB: {}", collection),
                db_type: DatabaseType::ChromaDB,
                path: format!("{}:{}/{}", host, port, collection),
                record_count: count,
                columns,
                table_name: collection_info.name.clone(),
                connection_state: ConnectionState::Connected,
            },
            client,
            base_url,
            collection_id: collection_uuid,
            current_offset: std::cell::Cell::new(0),
        })
    }

    fn response_to_records(&self, response: ChromaGetResponse) -> Vec<Record> {
        let documents = response.documents.unwrap_or_default();
        let metadatas = response.metadatas.unwrap_or_default();

        response
            .ids
            .into_iter()
            .enumerate()
            .map(|(i, id)| {
                let mut record = Record::new();
                record.fields.insert("id".to_string(), id);

                // Add document content (full - truncation happens in UI)
                if let Some(Some(doc)) = documents.get(i) {
                    record.fields.insert("document".to_string(), doc.clone());
                }

                // Add metadata fields
                if let Some(Some(meta)) = metadatas.get(i) {
                    for (key, value) in meta {
                        let str_value = match value {
                            serde_json::Value::String(s) => s.clone(),
                            serde_json::Value::Number(n) => n.to_string(),
                            serde_json::Value::Bool(b) => b.to_string(),
                            serde_json::Value::Null => "null".to_string(),
                            _ => value.to_string(),
                        };
                        record.fields.insert(key.clone(), str_value);
                    }

                    // Extract timestamp if available
                    if let Some(ts) = meta.get("processed_at").and_then(|v| v.as_str()) {
                        record.timestamp = Some(ts.to_string());
                    }
                }

                // Ensure timestamp is set for pagination (ChromaDB uses offset-based)
                // Use a placeholder if no timestamp in metadata
                if record.timestamp.is_none() {
                    record.timestamp = Some(format!("offset_{}", i));
                }

                record
            })
            .collect()
    }
}

impl DatabaseSource for ChromaDbSource {
    fn info(&self) -> &DatabaseInfo {
        &self.info
    }

    fn info_mut(&mut self) -> &mut DatabaseInfo {
        &mut self.info
    }

    fn health_check(&mut self) -> bool {
        // Use the v2 heartbeat endpoint to check if ChromaDB is running
        // Extract host:port from base_url and use /api/v2/heartbeat
        let heartbeat_url = self
            .base_url
            .replace("/tenants/default_tenant/databases/default_database", "/heartbeat");

        match self.client.get(&heartbeat_url).send() {
            Ok(response) if response.status().is_success() => {
                self.info.connection_state = ConnectionState::Connected;
                true
            }
            Ok(response) => {
                self.info.connection_state = ConnectionState::Disconnected(format!(
                    "ChromaDB returned status {}",
                    response.status()
                ));
                false
            }
            Err(e) => {
                let msg = if e.is_connect() {
                    "ChromaDB not running".to_string()
                } else if e.is_timeout() {
                    "ChromaDB connection timeout".to_string()
                } else {
                    format!("ChromaDB error: {}", e)
                };
                self.info.connection_state = ConnectionState::Disconnected(msg);
                false
            }
        }
    }

    fn fetch_recent(&self, limit: usize) -> Result<Vec<Record>, DbError> {
        // Reset pagination offset when fetching fresh
        self.current_offset.set(0);

        let url = format!("{}/collections/{}/get", self.base_url, self.collection_id);

        let request = ChromaGetRequest {
            limit: Some(limit),
            offset: None,
            include: vec!["documents".to_string(), "metadatas".to_string()],
        };

        let response: ChromaGetResponse = self
            .client
            .post(&url)
            .json(&request)
            .send()?
            .json()
            .map_err(|e| DbError::Query(format!("Failed to parse response: {}", e)))?;

        let records = self.response_to_records(response);

        // Update offset for next pagination call
        self.current_offset.set(records.len());

        Ok(records)
    }

    fn fetch_before(&self, _before: &str, limit: usize) -> Result<Vec<Record>, DbError> {
        // ChromaDB uses offset-based pagination
        let current = self.current_offset.get();

        // Don't fetch if we've reached the end
        if current >= self.info.record_count {
            return Ok(Vec::new());
        }

        let url = format!("{}/collections/{}/get", self.base_url, self.collection_id);

        let request = ChromaGetRequest {
            limit: Some(limit),
            offset: Some(current),
            include: vec!["documents".to_string(), "metadatas".to_string()],
        };

        let response: ChromaGetResponse = self
            .client
            .post(&url)
            .json(&request)
            .send()?
            .json()
            .map_err(|e| DbError::Query(format!("Failed to parse response: {}", e)))?;

        let records = self.response_to_records(response);

        // Update offset for next call
        self.current_offset.set(current + records.len());

        Ok(records)
    }

    fn execute_query(&self, _query: &str) -> Result<Vec<Record>, DbError> {
        // Note: ChromaDB v2 API requires actual embeddings for semantic search.
        // For now, just fetch recent documents. A future enhancement could add
        // local embedding generation using rust-bert or candle.
        //
        // TODO: Add local embedding model for semantic search support
        self.fetch_recent(50)
    }

    fn refresh_count(&mut self) -> Result<usize, DbError> {
        let count_url = format!(
            "{}/collections/{}/count",
            self.base_url, self.collection_id
        );
        let count: usize = self.client.get(&count_url).send()?.json()?;
        self.info.record_count = count;
        Ok(count)
    }

    fn clear_database(&mut self) -> Result<usize, DbError> {
        let count = self.info.record_count;

        if count == 0 {
            return Ok(0);
        }

        // Get all IDs first
        let get_url = format!("{}/collections/{}/get", self.base_url, self.collection_id);
        let request = ChromaGetRequest {
            limit: None, // Get all
            offset: None,
            include: vec![], // We only need IDs
        };

        let response: ChromaGetResponse = self
            .client
            .post(&get_url)
            .json(&request)
            .send()?
            .json()
            .map_err(|e| DbError::Query(format!("Failed to get IDs: {}", e)))?;

        if response.ids.is_empty() {
            self.info.record_count = 0;
            return Ok(0);
        }

        // Delete all IDs
        let delete_url = format!("{}/collections/{}/delete", self.base_url, self.collection_id);
        let delete_request = ChromaDeleteRequest {
            ids: response.ids,
        };

        self.client
            .post(&delete_url)
            .json(&delete_request)
            .send()
            .map_err(|e| DbError::Query(format!("Failed to delete: {}", e)))?;

        self.info.record_count = 0;
        Ok(count)
    }
}
