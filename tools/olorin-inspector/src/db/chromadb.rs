//! ChromaDB HTTP client implementation

use std::collections::HashMap;

use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

use super::traits::{DatabaseInfo, DatabaseSource, DatabaseType, DbError, Record};

#[derive(Serialize)]
struct ChromaGetRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    limit: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    offset: Option<usize>,
    include: Vec<String>,
}

#[derive(Serialize)]
struct ChromaQueryRequest {
    query_texts: Vec<String>,
    n_results: usize,
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
struct ChromaQueryResponse {
    ids: Vec<Vec<String>>,
    #[serde(default)]
    documents: Option<Vec<Vec<Option<String>>>>,
    #[serde(default)]
    metadatas: Option<Vec<Vec<Option<HashMap<String, serde_json::Value>>>>>,
    #[serde(default)]
    distances: Option<Vec<Vec<f64>>>,
}

#[derive(Deserialize, Debug)]
struct ChromaCollectionInfo {
    name: String,
    #[serde(default)]
    metadata: Option<HashMap<String, serde_json::Value>>,
}

/// ChromaDB database source via HTTP API
pub struct ChromaDbSource {
    info: DatabaseInfo,
    client: Client,
    base_url: String,
    collection_id: String,
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

        let base_url = format!("http://{}:{}/api/v1", host, port);

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

        // Get count
        let count_url = format!("{}/collections/{}/count", base_url, collection);
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
            },
            client,
            base_url,
            collection_id: collection.to_string(),
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

                // Add document content (truncated)
                if let Some(Some(doc)) = documents.get(i) {
                    let truncated: String = doc.chars().take(150).collect();
                    record
                        .fields
                        .insert("document".to_string(), truncated + "...");
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

                record
            })
            .collect()
    }
}

impl DatabaseSource for ChromaDbSource {
    fn info(&self) -> &DatabaseInfo {
        &self.info
    }

    fn fetch_recent(&self, limit: usize) -> Result<Vec<Record>, DbError> {
        // ChromaDB doesn't have native ordering, so we fetch and return as-is
        // The IDs are typically timestamp-based so newer entries come later
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

        let mut records = self.response_to_records(response);

        // Sort by timestamp (descending) if available
        records.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

        Ok(records)
    }

    fn fetch_before(&self, _before: &str, limit: usize) -> Result<Vec<Record>, DbError> {
        // ChromaDB doesn't support timestamp-based pagination natively
        // For now, just fetch with offset (this is a limitation)
        let url = format!("{}/collections/{}/get", self.base_url, self.collection_id);

        let request = ChromaGetRequest {
            limit: Some(limit),
            offset: Some(limit), // Use limit as offset for simplicity
            include: vec!["documents".to_string(), "metadatas".to_string()],
        };

        let response: ChromaGetResponse = self
            .client
            .post(&url)
            .json(&request)
            .send()?
            .json()
            .map_err(|e| DbError::Query(format!("Failed to parse response: {}", e)))?;

        let mut records = self.response_to_records(response);
        records.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

        Ok(records)
    }

    fn execute_query(&self, query: &str) -> Result<Vec<Record>, DbError> {
        // Treat the query as a semantic search query
        let url = format!("{}/collections/{}/query", self.base_url, self.collection_id);

        let request = ChromaQueryRequest {
            query_texts: vec![query.to_string()],
            n_results: 20,
            include: vec![
                "documents".to_string(),
                "metadatas".to_string(),
                "distances".to_string(),
            ],
        };

        let response: ChromaQueryResponse = self
            .client
            .post(&url)
            .json(&request)
            .send()?
            .json()
            .map_err(|e| DbError::Query(format!("Failed to parse response: {}", e)))?;

        // Flatten the nested response (query returns nested arrays)
        let mut records = Vec::new();

        if let Some(ids) = response.ids.first() {
            let documents = response.documents.and_then(|d| d.into_iter().next());
            let metadatas = response.metadatas.and_then(|m| m.into_iter().next());
            let distances = response.distances.and_then(|d| d.into_iter().next());

            for (i, id) in ids.iter().enumerate() {
                let mut record = Record::new();
                record.fields.insert("id".to_string(), id.clone());

                // Add distance
                if let Some(ref dists) = distances {
                    if let Some(dist) = dists.get(i) {
                        record
                            .fields
                            .insert("distance".to_string(), format!("{:.4}", dist));
                    }
                }

                // Add document
                if let Some(ref docs) = documents {
                    if let Some(Some(doc)) = docs.get(i) {
                        let truncated: String = doc.chars().take(150).collect();
                        record
                            .fields
                            .insert("document".to_string(), truncated + "...");
                    }
                }

                // Add metadata
                if let Some(ref metas) = metadatas {
                    if let Some(Some(meta)) = metas.get(i) {
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
                    }
                }

                records.push(record);
            }
        }

        Ok(records)
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
