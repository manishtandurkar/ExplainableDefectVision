import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './App.css';

const DEFECT_INFO = {
  'Crazing': {
    description: "Surface cracks caused by thermal stress or uneven cooling during the manufacturing process.",
    severity: "High",
    color: "#ef4444"
  },
  'Inclusion': {
    description: "Non-metallic impurities embedded in the steel matrix during the rolling or casting process.",
    severity: "Medium",
    color: "#f59e0b"
  },
  'Patches': {
    description: "Irregular surface texture variations caused by inconsistent coating or material composition.",
    severity: "Low",
    color: "#10b981"
  },
  'Pitted': {
    description: "Localized corrosion pits formed due to chemical contamination or environmental exposure.",
    severity: "High",
    color: "#ef4444"
  },
  'Rolled': {
    description: "Oxide scale particles trapped and embedded in the surface during hot rolling operations.",
    severity: "Medium",
    color: "#f59e0b"
  },
  'Scratches': {
    description: "Linear mechanical abrasions caused by contact with tools, handling equipment, or processing machinery.",
    severity: "Low",
    color: "#10b981"
  }
};

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.bmp']
    },
    maxFiles: 1
  });

  const analyzeImage = async () => {
    if (!image) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', image);

    try {
      // Replace with your actual API endpoint
      const response = await axios.post('http://localhost:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to analyze image. Make sure the backend server is running.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const resetAnalysis = () => {
    setImage(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const chartData = result?.probabilities ? 
    Object.entries(result.probabilities).map(([name, prob]) => ({
      name,
      probability: (prob * 100).toFixed(1)
    })) : [];

  return (
    <div className="App">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <h1>🏭 Explainable Defect Detection</h1>
          <p>Vision Transformer-based Industrial Surface Defect Analysis</p>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        <div className="container">
          
          {/* Upload Section */}
          <div className="upload-section">
            <h2>📤 Upload Steel Surface Image</h2>
            
            {!preview ? (
              <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
                <input {...getInputProps()} />
                <div className="dropzone-content">
                  <svg className="upload-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                  <p className="dropzone-text">
                    {isDragActive ? 'Drop image here...' : 'Drag & drop an image, or click to select'}
                  </p>
                  <p className="dropzone-hint">Supports: PNG, JPG, JPEG, BMP</p>
                </div>
              </div>
            ) : (
              <div className="preview-section">
                <img src={preview} alt="Preview" className="preview-image" />
                <div className="preview-actions">
                  <button onClick={analyzeImage} disabled={loading} className="btn btn-primary">
                    {loading ? (
                      <>
                        <span className="spinner"></span>
                        Analyzing...
                      </>
                    ) : (
                      <>🔍 Analyze Defect</>
                    )}
                  </button>
                  <button onClick={resetAnalysis} className="btn btn-secondary">
                    🔄 Upload New Image
                  </button>
                </div>
              </div>
            )}

            {error && (
              <div className="alert alert-error">
                <strong>⚠️ Error:</strong> {error}
              </div>
            )}
          </div>

          {/* Results Section */}
          {result && (
            <div className="results-section">
              <h2>🎯 Analysis Results</h2>

              {/* Prediction Card */}
              <div className="prediction-card">
                <div className="prediction-header">
                  <h3>Detected Defect</h3>
                  <span className={`severity-badge severity-${DEFECT_INFO[result.predicted_class]?.severity.toLowerCase()}`}>
                    {DEFECT_INFO[result.predicted_class]?.severity}
                  </span>
                </div>
                <div className="prediction-body">
                  <div className="defect-name">{result.predicted_class}</div>
                  <div className="confidence">
                    Confidence: <strong>{(result.confidence * 100).toFixed(2)}%</strong>
                  </div>
                  <div className="defect-description">
                    💡 <strong>Root Cause:</strong> {DEFECT_INFO[result.predicted_class]?.description}
                  </div>
                </div>
              </div>

              {/* Probability Chart */}
              <div className="chart-card">
                <h3>📊 Class Probabilities</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
                    <YAxis label={{ value: 'Probability (%)', angle: -90, position: 'insideLeft' }} />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="probability" fill="#3b82f6" />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Heatmap Visualization */}
              {result.heatmap && (
                <div className="heatmap-card">
                  <h3>🔍 Attention Visualization</h3>
                  <img 
                    src={`data:image/png;base64,${result.heatmap}`} 
                    alt="Attention Heatmap" 
                    className="heatmap-image"
                  />
                  <p className="heatmap-caption">
                    Red areas indicate regions the model focused on for classification
                  </p>
                </div>
              )}

              {/* Detailed Info */}
              <div className="info-card">
                <h3>📋 Detailed Analysis</h3>
                <div className="info-grid">
                  <div className="info-item">
                    <span className="info-label">Model:</span>
                    <span className="info-value">Vision Transformer (ViT)</span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">Class ID:</span>
                    <span className="info-value">{result.class_id}</span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">Processing Time:</span>
                    <span className="info-value">{result.processing_time ? `${result.processing_time.toFixed(2)}s` : 'N/A'}</span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">Image Size:</span>
                    <span className="info-value">{result.image_size || '224x224'}</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Info Section */}
          <div className="info-section">
            <h3>ℹ️ About This Tool</h3>
            <div className="info-grid-cards">
              <div className="info-card-small">
                <h4>🎯 Accuracy</h4>
                <p>100% validation accuracy on all 6 defect classes</p>
              </div>
              <div className="info-card-small">
                <h4>🤖 Technology</h4>
                <p>Vision Transformer with Grad-CAM explainability</p>
              </div>
              <div className="info-card-small">
                <h4>📊 Dataset</h4>
                <p>Trained on NEU Surface Defect Database</p>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p>🏭 Explainable Multi-Class Industrial Defect Detection | Built with React & PyTorch</p>
      </footer>
    </div>
  );
}

export default App;
