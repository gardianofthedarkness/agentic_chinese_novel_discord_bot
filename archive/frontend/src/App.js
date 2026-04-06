import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    if (!query.trim()) return;
    
    setLoading(true);
    try {
      const response = await axios.post('http://localhost:5000/search', {
        query,
        top_k: 5
      });
      setResults(response.data);
    } catch (error) {
      console.error('Error searching:', error);
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>MoJin RAG Search</h1>
      <div className="search-box">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter your query..."
          disabled={loading}
        />
        <button onClick={handleSearch} disabled={loading}>
          {loading ? 'Searching...' : 'Search'}
        </button>
      </div>
      
      <div className="results">
        {results.length > 0 ? (
          results.map((result, index) => (
            <div key={index} className="result-card">
              <p className="score">Score: {result.score.toFixed(4)}</p>
              <p className="chunk">{result.chunk}</p>
              <p className="coordinates">Coordinates: {result.coordinate.join(', ')}</p>
            </div>
          ))
        ) : (
          <p>No results found</p>
        )}
      </div>
    </div>
  );
}

export default App;
