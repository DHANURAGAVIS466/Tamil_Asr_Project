'use client';

import React, { useState, useEffect } from 'react';
import { Mic, BarChart2, Database, Layers, Sparkles, Send, PlayCircle, RefreshCw } from 'lucide-react';
import Header from './components/Header';
import AudioRecorder from './components/AudioRecorder';
import FileUpload from './components/FileUpload';
import SampleSelector from './components/SampleSelector';
import TranscriptionResults from './components/TranscriptionResults';
import MetricsOverview from './components/MetricsOverview';
import DatasetExplorer from './components/DatasetExplorer';
import RoadmapViewer from './components/RoadmapViewer';

export default function Home() {
  const [activeTab, setActiveTab] = useState('studio');
  const [serverStatus, setServerStatus] = useState(null);
  const [samples, setSamples] = useState([]);
  const [datasetStats, setDatasetStats] = useState(null);

  const [currentAudioBlob, setCurrentAudioBlob] = useState(null);
  const [currentAudioName, setCurrentAudioName] = useState('');
  const [selectedSample, setSelectedSample] = useState(null);
  const [referenceText, setReferenceText] = useState('');
  
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [transcriptionResults, setTranscriptionResults] = useState(null);
  const [transcriptionError, setTranscriptionError] = useState(null);

  // Helper to fetch with direct fallback
  const apiFetch = async (endpoint, options = {}) => {
    try {
      const res = await fetch(endpoint, options);
      if (res.ok) return res;
      // If error or proxy failure, attempt direct backend port 8000
      if (!endpoint.startsWith('http')) {
        const directUrl = `http://127.0.0.1:8000${endpoint.startsWith('/') ? endpoint : '/' + endpoint}`;
        const directRes = await fetch(directUrl, options);
        return directRes;
      }
      return res;
    } catch (err) {
      if (!endpoint.startsWith('http')) {
        const directUrl = `http://127.0.0.1:8000${endpoint.startsWith('/') ? endpoint : '/' + endpoint}`;
        return await fetch(directUrl, options);
      }
      throw err;
    }
  };

  // Fetch initial data
  const fetchData = async () => {
    try {
      const [healthRes, samplesRes, statsRes] = await Promise.all([
        apiFetch('/api/health').then(r => r.ok ? r.json() : null).catch(() => null),
        apiFetch('/api/samples').then(r => r.ok ? r.json() : null).catch(() => null),
        apiFetch('/api/dataset-stats').then(r => r.ok ? r.json() : null).catch(() => null),
      ]);

      if (healthRes) setServerStatus(healthRes);
      if (samplesRes?.samples) {
        setSamples(samplesRes.samples);
      }
      if (statsRes) setDatasetStats(statsRes);
    } catch (e) {
      console.error('Error connecting to backend:', e);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleAudioReady = (blob, name) => {
    setCurrentAudioBlob(blob);
    setCurrentAudioName(name);
    setSelectedSample(null);
    setTranscriptionError(null);
  };

  const handleSelectSample = async (sample) => {
    setSelectedSample(sample);
    setReferenceText(sample.reference);
    setCurrentAudioName(sample.title);
    setTranscriptionError(null);

    try {
      const res = await apiFetch(sample.audio_url);
      const blob = await res.blob();
      setCurrentAudioBlob(blob);
    } catch (err) {
      console.error('Failed to load sample audio:', err);
    }
  };

  const handleTranscribe = async () => {
    if (!currentAudioBlob) {
      setTranscriptionError('Please record audio, select a preset sample, or upload a file first.');
      return;
    }

    setIsTranscribing(true);
    setTranscriptionResults(null);
    setTranscriptionError(null);

    try {
      const formData = new FormData();
      formData.append('file', currentAudioBlob, currentAudioName || 'microphone_recording.webm');
      if (referenceText) {
        formData.append('reference', referenceText);
      }

      const res = await apiFetch('/api/transcribe', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        let errText = 'Transcription failed';
        try {
          const errData = await res.json();
          errText = errData.detail || errData.message || JSON.stringify(errData);
        } catch {
          const raw = await res.text();
          errText = raw || `Server responded with status ${res.status}`;
        }
        throw new Error(errText);
      }

      const data = await res.json();
      setTranscriptionResults(data);
    } catch (err) {
      console.error('Transcription error:', err);
      setTranscriptionError(err.message || 'Error occurred during speech recognition inference.');
    } finally {
      setIsTranscribing(false);
    }
  };

  return (
    <main>
      <Header serverStatus={serverStatus} />

      {/* Navigation Tabs */}
      <nav className="tabs-nav">
        <button
          className={`tab-btn ${activeTab === 'studio' ? 'active' : ''}`}
          onClick={() => setActiveTab('studio')}
        >
          <Mic size={16} /> Live Tamil Studio
        </button>

        <button
          className={`tab-btn ${activeTab === 'benchmark' ? 'active' : ''}`}
          onClick={() => setActiveTab('benchmark')}
        >
          <BarChart2 size={16} /> Model Benchmark
        </button>

        <button
          className={`tab-btn ${activeTab === 'dataset' ? 'active' : ''}`}
          onClick={() => setActiveTab('dataset')}
        >
          <Database size={16} /> OpenSLR-65 Explorer
        </button>

        <button
          className={`tab-btn ${activeTab === 'roadmap' ? 'active' : ''}`}
          onClick={() => setActiveTab('roadmap')}
        >
          <Layers size={16} /> Research Roadmap (Phases 1-7)
        </button>
      </nav>

      {/* Tab Content */}
      {activeTab === 'studio' && (
        <div className="studio-grid">
          {/* Left Column: Audio Input Sources */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            <AudioRecorder
              onAudioReady={handleAudioReady}
              isTranscribing={isTranscribing}
            />

            <FileUpload
              onFileSelect={handleAudioReady}
              isTranscribing={isTranscribing}
            />

            <SampleSelector
              samples={samples}
              selectedSample={selectedSample}
              onSelectSample={handleSelectSample}
              isTranscribing={isTranscribing}
            />
          </div>

          {/* Right Column: Execution Control & Results */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            {/* Action Bar */}
            <div
              className="glass-card"
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '18px 24px',
                flexWrap: 'wrap',
                gap: '14px',
              }}
            >
              <div>
                <h4 style={{ fontSize: '16px', fontWeight: 700 }}>
                  {currentAudioName ? `Selected: ${currentAudioName}` : 'Ready for Transcription'}
                </h4>
                <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '2px' }}>
                  {currentAudioBlob ? 'Audio captured and ready for inference' : 'Pick a sample, record, or upload audio'}
                </p>
              </div>

              <button
                onClick={handleTranscribe}
                className="btn-primary"
                disabled={!currentAudioBlob || isTranscribing}
                style={{
                  opacity: !currentAudioBlob || isTranscribing ? 0.6 : 1,
                  cursor: !currentAudioBlob || isTranscribing ? 'not-allowed' : 'pointer',
                }}
              >
                {isTranscribing ? (
                  <>
                    <RefreshCw size={18} className="animate-spin" /> Transcribing...
                  </>
                ) : (
                  <>
                    <Sparkles size={18} /> Transcribe Speech (Whisper & Wav2Vec2)
                  </>
                )}
              </button>
            </div>

            {transcriptionError && (
              <div
                style={{
                  background: 'rgba(239, 68, 68, 0.12)',
                  border: '1px solid rgba(239, 68, 68, 0.35)',
                  borderRadius: 'var(--radius-md)',
                  padding: '14px 18px',
                  color: '#fca5a5',
                  fontSize: '14px',
                  lineHeight: '1.5',
                }}
              >
                <div style={{ fontWeight: 600, color: '#f87171', marginBottom: '4px' }}>
                  ⚠️ Transcription Issue
                </div>
                <div>{transcriptionError}</div>
              </div>
            )}

            {/* Transcription Output */}
            <TranscriptionResults
              results={transcriptionResults}
              isTranscribing={isTranscribing}
            />
          </div>
        </div>
      )}

      {activeTab === 'benchmark' && (
        <MetricsOverview datasetStats={datasetStats} />
      )}

      {activeTab === 'dataset' && (
        <DatasetExplorer datasetStats={datasetStats} />
      )}

      {activeTab === 'roadmap' && (
        <RoadmapViewer />
      )}
    </main>
  );
}
