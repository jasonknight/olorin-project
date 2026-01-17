#!/usr/bin/env python3
"""
Tests for the search tool server.

Includes unit tests for internal functions and integration tests for API endpoints.
"""

import json
import math
import sys
from pathlib import Path


# Add tool directory to path
TOOL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(TOOL_DIR))


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_ok_when_connected(self, client, mock_chroma_client):
        """Test health endpoint returns ok when ChromaDB is connected."""
        response = client.get("/health")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "ok"

    def test_health_returns_error_when_heartbeat_fails(
        self, client, mock_chroma_client
    ):
        """Test health endpoint returns error when ChromaDB heartbeat fails."""
        mock_chroma_client.heartbeat.side_effect = Exception("Connection failed")
        response = client.get("/health")
        assert response.status_code == 503
        data = json.loads(response.data)
        assert data["status"] == "error"


class TestDescribeEndpoint:
    """Tests for the /describe endpoint."""

    def test_describe_returns_tool_metadata(self, client):
        """Test describe endpoint returns correct tool metadata."""
        response = client.get("/describe")
        assert response.status_code == 200
        data = json.loads(response.data)

        assert data["name"] == "search"
        assert "description" in data
        assert "parameters" in data

    def test_describe_includes_all_parameters(self, client):
        """Test describe endpoint includes all expected parameters."""
        response = client.get("/describe")
        data = json.loads(response.data)

        param_names = [p["name"] for p in data["parameters"]]
        assert "query" in param_names
        assert "page" in param_names
        assert "per_page" in param_names
        assert "collection" in param_names
        assert "include_distances" in param_names

    def test_describe_query_is_required(self, client):
        """Test that query parameter is marked as required."""
        response = client.get("/describe")
        data = json.loads(response.data)

        query_param = next(p for p in data["parameters"] if p["name"] == "query")
        assert query_param["required"] is True

    def test_describe_optional_params_not_required(self, client):
        """Test that optional parameters are not required."""
        response = client.get("/describe")
        data = json.loads(response.data)

        optional_params = ["page", "per_page", "collection", "include_distances"]
        for param in data["parameters"]:
            if param["name"] in optional_params:
                assert param["required"] is False


class TestCallEndpointValidation:
    """Tests for /call endpoint input validation."""

    def test_call_requires_json_body(self, client):
        """Test call endpoint requires JSON body."""
        response = client.post("/call", data="not json")
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data["success"] is False
        assert data["error"]["type"] == "ValidationError"

    def test_call_requires_query_parameter(self, initialized_client):
        """Test call endpoint requires query parameter."""
        client, _, _ = initialized_client
        response = client.post(
            "/call", data=json.dumps({"page": 1}), content_type="application/json"
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data["success"] is False
        assert "query" in data["error"]["message"].lower()

    def test_call_query_must_be_string(self, initialized_client):
        """Test call endpoint requires query to be a string."""
        client, _, _ = initialized_client
        response = client.post(
            "/call", data=json.dumps({"query": 123}), content_type="application/json"
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data["success"] is False

    def test_call_empty_query_rejected(self, initialized_client):
        """Test call endpoint rejects empty query string."""
        client, _, _ = initialized_client
        response = client.post(
            "/call", data=json.dumps({"query": ""}), content_type="application/json"
        )
        assert response.status_code == 400


class TestCallEndpointDefaults:
    """Tests for /call endpoint default values."""

    def test_default_page_is_one(self, initialized_client, sample_query_results):
        """Test default page is 1."""
        client, mock_chroma, _ = initialized_client
        mock_chroma.get_collection().query.return_value = sample_query_results
        mock_chroma.get_collection().count.return_value = 100

        response = client.post(
            "/call",
            data=json.dumps({"query": "test"}),
            content_type="application/json",
        )
        data = json.loads(response.data)
        assert data["result"]["page"] == 1

    def test_default_per_page_is_ten(self, initialized_client, sample_query_results):
        """Test default per_page is 10."""
        client, mock_chroma, _ = initialized_client
        mock_chroma.get_collection().query.return_value = sample_query_results
        mock_chroma.get_collection().count.return_value = 100

        response = client.post(
            "/call",
            data=json.dumps({"query": "test"}),
            content_type="application/json",
        )
        data = json.loads(response.data)
        assert data["result"]["per_page"] == 10

    def test_default_collection_is_documents(
        self, initialized_client, sample_query_results
    ):
        """Test default collection is 'documents'."""
        client, mock_chroma, _ = initialized_client
        mock_chroma.get_collection().query.return_value = sample_query_results
        mock_chroma.get_collection().count.return_value = 100

        client.post(
            "/call",
            data=json.dumps({"query": "test"}),
            content_type="application/json",
        )
        mock_chroma.get_collection.assert_called_with(name="documents")

    def test_default_include_distances_is_true(
        self, initialized_client, sample_query_results
    ):
        """Test default include_distances is True."""
        client, mock_chroma, _ = initialized_client
        mock_chroma.get_collection().query.return_value = sample_query_results
        mock_chroma.get_collection().count.return_value = 5

        response = client.post(
            "/call",
            data=json.dumps({"query": "test"}),
            content_type="application/json",
        )
        data = json.loads(response.data)
        # Check that results include distance field
        if data["result"]["results"]:
            assert "distance" in data["result"]["results"][0]


class TestCallEndpointSearch:
    """Tests for /call endpoint search functionality."""

    def test_successful_search(self, initialized_client, sample_query_results):
        """Test successful search returns results."""
        client, mock_chroma, _ = initialized_client
        mock_chroma.get_collection().query.return_value = sample_query_results
        mock_chroma.get_collection().count.return_value = 5

        response = client.post(
            "/call",
            data=json.dumps({"query": "test search"}),
            content_type="application/json",
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert "result" in data

    def test_search_returns_query_in_response(
        self, initialized_client, sample_query_results
    ):
        """Test search response includes the original query."""
        client, mock_chroma, _ = initialized_client
        mock_chroma.get_collection().query.return_value = sample_query_results
        mock_chroma.get_collection().count.return_value = 5

        response = client.post(
            "/call",
            data=json.dumps({"query": "my test query"}),
            content_type="application/json",
        )
        data = json.loads(response.data)
        assert data["result"]["query"] == "my test query"

    def test_search_returns_total_results(
        self, initialized_client, sample_query_results
    ):
        """Test search response includes total_results."""
        client, mock_chroma, _ = initialized_client
        mock_chroma.get_collection().query.return_value = sample_query_results
        mock_chroma.get_collection().count.return_value = 150

        response = client.post(
            "/call",
            data=json.dumps({"query": "test"}),
            content_type="application/json",
        )
        data = json.loads(response.data)
        assert data["result"]["total_results"] == 150

    def test_search_returns_result_items(
        self, initialized_client, sample_query_results
    ):
        """Test search response includes result items with expected fields."""
        client, mock_chroma, _ = initialized_client
        mock_chroma.get_collection().query.return_value = sample_query_results
        mock_chroma.get_collection().count.return_value = 5

        response = client.post(
            "/call",
            data=json.dumps({"query": "test"}),
            content_type="application/json",
        )
        data = json.loads(response.data)

        results = data["result"]["results"]
        assert len(results) > 0

        first_result = results[0]
        assert "id" in first_result
        assert "text" in first_result
        assert "metadata" in first_result
        assert "distance" in first_result

    def test_search_without_distances(self, initialized_client, sample_query_results):
        """Test search with include_distances=False omits distances."""
        client, mock_chroma, _ = initialized_client
        mock_chroma.get_collection().query.return_value = sample_query_results
        mock_chroma.get_collection().count.return_value = 5

        response = client.post(
            "/call",
            data=json.dumps({"query": "test", "include_distances": False}),
            content_type="application/json",
        )
        data = json.loads(response.data)

        if data["result"]["results"]:
            assert "distance" not in data["result"]["results"][0]

    def test_empty_collection_returns_zero_results(self, initialized_client):
        """Test search on empty collection returns zero results."""
        client, mock_chroma, _ = initialized_client
        mock_chroma.get_collection().count.return_value = 0

        response = client.post(
            "/call",
            data=json.dumps({"query": "test"}),
            content_type="application/json",
        )
        data = json.loads(response.data)
        assert data["success"] is True
        assert data["result"]["total_results"] == 0
        assert data["result"]["results"] == []


class TestPagination:
    """Tests for pagination functionality."""

    def test_total_pages_calculation(self, initialized_client, sample_query_results):
        """Test total_pages is calculated correctly."""
        client, mock_chroma, _ = initialized_client
        mock_chroma.get_collection().query.return_value = sample_query_results
        mock_chroma.get_collection().count.return_value = 25

        response = client.post(
            "/call",
            data=json.dumps({"query": "test", "per_page": 10}),
            content_type="application/json",
        )
        data = json.loads(response.data)
        assert data["result"]["total_pages"] == 3  # ceil(25/10) = 3

    def test_total_pages_with_exact_division(
        self, initialized_client, sample_query_results
    ):
        """Test total_pages when total divides evenly by per_page."""
        client, mock_chroma, _ = initialized_client
        mock_chroma.get_collection().query.return_value = sample_query_results
        mock_chroma.get_collection().count.return_value = 20

        response = client.post(
            "/call",
            data=json.dumps({"query": "test", "per_page": 10}),
            content_type="application/json",
        )
        data = json.loads(response.data)
        assert data["result"]["total_pages"] == 2  # 20/10 = 2

    def test_page_number_in_response(self, initialized_client, sample_query_results):
        """Test requested page number is returned in response."""
        client, mock_chroma, _ = initialized_client
        mock_chroma.get_collection().query.return_value = sample_query_results
        mock_chroma.get_collection().count.return_value = 100

        response = client.post(
            "/call",
            data=json.dumps({"query": "test", "page": 3}),
            content_type="application/json",
        )
        data = json.loads(response.data)
        assert data["result"]["page"] == 3

    def test_per_page_in_response(self, initialized_client, sample_query_results):
        """Test requested per_page is returned in response."""
        client, mock_chroma, _ = initialized_client
        mock_chroma.get_collection().query.return_value = sample_query_results
        mock_chroma.get_collection().count.return_value = 100

        response = client.post(
            "/call",
            data=json.dumps({"query": "test", "per_page": 5}),
            content_type="application/json",
        )
        data = json.loads(response.data)
        assert data["result"]["per_page"] == 5

    def test_first_page_returns_first_results(
        self, initialized_client, large_query_results
    ):
        """Test first page returns first N results."""
        client, mock_chroma, _ = initialized_client
        mock_chroma.get_collection().query.return_value = large_query_results
        mock_chroma.get_collection().count.return_value = 25

        response = client.post(
            "/call",
            data=json.dumps({"query": "test", "page": 1, "per_page": 5}),
            content_type="application/json",
        )
        data = json.loads(response.data)

        results = data["result"]["results"]
        assert len(results) == 5
        assert results[0]["id"] == "doc0"
        assert results[4]["id"] == "doc4"

    def test_second_page_returns_next_results(
        self, initialized_client, large_query_results
    ):
        """Test second page returns correct slice of results."""
        client, mock_chroma, _ = initialized_client
        mock_chroma.get_collection().query.return_value = large_query_results
        mock_chroma.get_collection().count.return_value = 25

        response = client.post(
            "/call",
            data=json.dumps({"query": "test", "page": 2, "per_page": 5}),
            content_type="application/json",
        )
        data = json.loads(response.data)

        results = data["result"]["results"]
        assert len(results) == 5
        assert results[0]["id"] == "doc5"
        assert results[4]["id"] == "doc9"

    def test_third_page_returns_correct_results(
        self, initialized_client, large_query_results
    ):
        """Test third page returns correct slice of results."""
        client, mock_chroma, _ = initialized_client
        mock_chroma.get_collection().query.return_value = large_query_results
        mock_chroma.get_collection().count.return_value = 25

        response = client.post(
            "/call",
            data=json.dumps({"query": "test", "page": 3, "per_page": 5}),
            content_type="application/json",
        )
        data = json.loads(response.data)

        results = data["result"]["results"]
        assert len(results) == 5
        assert results[0]["id"] == "doc10"
        assert results[4]["id"] == "doc14"

    def test_last_page_partial_results(self, initialized_client, large_query_results):
        """Test last page returns partial results when not full."""
        client, mock_chroma, _ = initialized_client
        # 25 results, 5 per page = 5 pages, last page has 5 results
        # Let's use 23 results instead for a partial last page
        results_23 = {
            "ids": [[f"doc{i}" for i in range(23)]],
            "documents": [[f"Document {i} content" for i in range(23)]],
            "metadatas": [
                [{"source": f"/path/file{i}.md", "h1": f"Title {i}"} for i in range(23)]
            ],
            "distances": [[0.1 + (i * 0.01) for i in range(23)]],
        }
        mock_chroma.get_collection().query.return_value = results_23
        mock_chroma.get_collection().count.return_value = 23

        response = client.post(
            "/call",
            data=json.dumps({"query": "test", "page": 5, "per_page": 5}),
            content_type="application/json",
        )
        data = json.loads(response.data)

        results = data["result"]["results"]
        assert len(results) == 3  # 23 - (4*5) = 3
        assert data["result"]["total_pages"] == 5  # ceil(23/5) = 5

    def test_page_beyond_results_returns_empty(
        self, initialized_client, sample_query_results
    ):
        """Test requesting page beyond total returns empty results."""
        client, mock_chroma, _ = initialized_client
        mock_chroma.get_collection().query.return_value = sample_query_results
        mock_chroma.get_collection().count.return_value = 5

        response = client.post(
            "/call",
            data=json.dumps({"query": "test", "page": 10, "per_page": 5}),
            content_type="application/json",
        )
        data = json.loads(response.data)
        assert data["result"]["results"] == []

    def test_invalid_page_defaults_to_one(
        self, initialized_client, sample_query_results
    ):
        """Test invalid page number defaults to 1."""
        client, mock_chroma, _ = initialized_client
        mock_chroma.get_collection().query.return_value = sample_query_results
        mock_chroma.get_collection().count.return_value = 5

        response = client.post(
            "/call",
            data=json.dumps({"query": "test", "page": -1}),
            content_type="application/json",
        )
        data = json.loads(response.data)
        assert data["result"]["page"] == 1

    def test_invalid_per_page_defaults(self, initialized_client, sample_query_results):
        """Test invalid per_page defaults to 10."""
        client, mock_chroma, _ = initialized_client
        mock_chroma.get_collection().query.return_value = sample_query_results
        mock_chroma.get_collection().count.return_value = 100

        response = client.post(
            "/call",
            data=json.dumps({"query": "test", "per_page": -5}),
            content_type="application/json",
        )
        data = json.loads(response.data)
        assert data["result"]["per_page"] == 10

    def test_per_page_capped_at_max(self, initialized_client, sample_query_results):
        """Test per_page is capped at maximum (100)."""
        client, mock_chroma, _ = initialized_client
        mock_chroma.get_collection().query.return_value = sample_query_results
        mock_chroma.get_collection().count.return_value = 1000

        response = client.post(
            "/call",
            data=json.dumps({"query": "test", "per_page": 500}),
            content_type="application/json",
        )
        data = json.loads(response.data)
        assert data["result"]["per_page"] == 100

    def test_different_per_page_values(self, initialized_client, large_query_results):
        """Test pagination works with different per_page values."""
        client, mock_chroma, _ = initialized_client
        mock_chroma.get_collection().query.return_value = large_query_results
        mock_chroma.get_collection().count.return_value = 25

        for per_page in [1, 3, 7, 10, 25]:
            response = client.post(
                "/call",
                data=json.dumps({"query": "test", "page": 1, "per_page": per_page}),
                content_type="application/json",
            )
            data = json.loads(response.data)
            assert data["result"]["per_page"] == per_page
            assert data["result"]["total_pages"] == math.ceil(25 / per_page)
            assert len(data["result"]["results"]) == min(per_page, 25)


class TestCollectionSelection:
    """Tests for collection selection."""

    def test_custom_collection(self, initialized_client, sample_query_results):
        """Test using a custom collection name."""
        client, mock_chroma, _ = initialized_client
        mock_chroma.get_collection().query.return_value = sample_query_results
        mock_chroma.get_collection().count.return_value = 5

        client.post(
            "/call",
            data=json.dumps({"query": "test", "collection": "my_collection"}),
            content_type="application/json",
        )
        mock_chroma.get_collection.assert_called_with(name="my_collection")

    def test_nonexistent_collection_returns_error(self, initialized_client):
        """Test requesting nonexistent collection returns error."""
        client, mock_chroma, _ = initialized_client
        mock_chroma.get_collection.side_effect = Exception("Collection not found")

        response = client.post(
            "/call",
            data=json.dumps({"query": "test", "collection": "nonexistent"}),
            content_type="application/json",
        )
        assert response.status_code == 404
        data = json.loads(response.data)
        assert data["success"] is False
        assert data["error"]["type"] == "NotFoundError"


class TestEmbeddingsIntegration:
    """Tests for embeddings tool integration."""

    def test_embeddings_called_with_query_mode(
        self, initialized_client, sample_query_results
    ):
        """Test embeddings tool is called with query mode."""
        client, mock_chroma, mock_post = initialized_client
        mock_chroma.get_collection().query.return_value = sample_query_results
        mock_chroma.get_collection().count.return_value = 5

        client.post(
            "/call",
            data=json.dumps({"query": "test query"}),
            content_type="application/json",
        )

        # Check embeddings was called with correct parameters
        mock_post.assert_called()
        call_args = mock_post.call_args
        assert call_args[1]["json"]["texts"] == ["test query"]
        assert call_args[1]["json"]["mode"] == "query"

    def test_embeddings_failure_returns_error(self, initialized_client):
        """Test embeddings tool failure returns appropriate error."""
        client, mock_chroma, mock_post = initialized_client
        mock_chroma.get_collection().count.return_value = 100

        # Simulate embeddings tool failure
        mock_post.return_value.status_code = 500

        response = client.post(
            "/call",
            data=json.dumps({"query": "test"}),
            content_type="application/json",
        )
        assert response.status_code == 503
        data = json.loads(response.data)
        assert data["success"] is False
        assert "embedding" in data["error"]["message"].lower()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_result(self, initialized_client):
        """Test handling of single result."""
        client, mock_chroma, _ = initialized_client
        single_result = {
            "ids": [["doc1"]],
            "documents": [["Single document"]],
            "metadatas": [[{"source": "/path/file.md"}]],
            "distances": [[0.1]],
        }
        mock_chroma.get_collection().query.return_value = single_result
        mock_chroma.get_collection().count.return_value = 1

        response = client.post(
            "/call",
            data=json.dumps({"query": "test"}),
            content_type="application/json",
        )
        data = json.loads(response.data)
        assert data["result"]["total_results"] == 1
        assert data["result"]["total_pages"] == 1
        assert len(data["result"]["results"]) == 1

    def test_unicode_query(self, initialized_client, sample_query_results):
        """Test handling of unicode characters in query."""
        client, mock_chroma, _ = initialized_client
        mock_chroma.get_collection().query.return_value = sample_query_results
        mock_chroma.get_collection().count.return_value = 5

        response = client.post(
            "/call",
            data=json.dumps({"query": "test 日本語 émoji 🎉"}),
            content_type="application/json",
        )
        data = json.loads(response.data)
        assert data["success"] is True
        assert data["result"]["query"] == "test 日本語 émoji 🎉"

    def test_very_long_query(self, initialized_client, sample_query_results):
        """Test handling of very long query string."""
        client, mock_chroma, _ = initialized_client
        mock_chroma.get_collection().query.return_value = sample_query_results
        mock_chroma.get_collection().count.return_value = 5

        long_query = "test " * 1000
        response = client.post(
            "/call",
            data=json.dumps({"query": long_query}),
            content_type="application/json",
        )
        data = json.loads(response.data)
        assert data["success"] is True

    def test_special_characters_in_query(
        self, initialized_client, sample_query_results
    ):
        """Test handling of special characters in query."""
        client, mock_chroma, _ = initialized_client
        mock_chroma.get_collection().query.return_value = sample_query_results
        mock_chroma.get_collection().count.return_value = 5

        response = client.post(
            "/call",
            data=json.dumps({"query": "test \"quotes\" and 'apostrophes' & symbols"}),
            content_type="application/json",
        )
        data = json.loads(response.data)
        assert data["success"] is True
