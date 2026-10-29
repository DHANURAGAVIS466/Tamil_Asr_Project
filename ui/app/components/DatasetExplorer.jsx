'use client';

import React, { useState } from 'react';
import { Database, Search, Download, Filter, FileText, CheckCircle2, UserCheck } from 'lucide-react';

export default function DatasetExplorer({ datasetStats }) {
  const [searchTerm, setSearchTerm] = useState('');
  const samples = datasetStats?.sample_transcriptions || [];

  const filteredSamples = samples.filter((s) => {
    return (
      s.reference.toLowerCase().includes(searchTerm.toLowerCase()) ||
      s.whisper.toLowerCase().includes(searchTerm.toLowerCase()) ||
      s.wav2vec.toLowerCase().includes(searchTerm.toLowerCase())
    );
  });

  const exportCSV = () => {
    const headers = ['id', 'reference', 'whisper_prediction', 'wav2vec_prediction', 'wer_whisper', 'cer_whisper', 'wer_wav2vec', 'cer_wav2vec'];
    const rows = samples.map(s => [
      s.id,
      `"${s.reference}"`,
      `"${s.whisper}"`,
      `"${s.wav2vec}"`,
      s.wer_whisper,
      s.cer_whisper,
      s.wer_wav2vec,
      s.cer_wav2vec
    ]);

    const csvContent = 'data:text/csv;charset=utf-8,\uFEFF' + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', 'tamil_asr_results.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>
      {/* Dataset Metadata Grid */}
      <div className="metrics-grid" style={{ marginTop: 0 }}>
        <div className="metric-box">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '13px', color: 'var(--text-secondary)', fontWeight: 600 }}>Corpus Name</span>
            <Database size={18} color="var(--accent-indigo)" />
          </div>
          <div style={{ fontSize: '20px', fontWeight: 800, color: 'var(--text-primary)', marginTop: '4px' }}>
            OpenSLR-65 Tamil
          </div>
          <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>deepdml/openslr65-tamil (Hugging Face)</p>
        </div>

        <div className="metric-box">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '13px', color: 'var(--text-secondary)', fontWeight: 600 }}>Total Speech</span>
            <FileText size={18} color="var(--accent-purple)" />
          </div>
          <div className="metric-val">7.08 hrs</div>
          <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>4,291 High-Quality Utterances</p>
        </div>

        <div className="metric-box">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '13px', color: 'var(--text-secondary)', fontWeight: 600 }}>Female Speakers</span>
            <UserCheck size={18} color="var(--accent-cyan)" />
          </div>
          <div className="metric-val" style={{ background: 'linear-gradient(135deg, #a5f3fc, #06b6d4)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            2,335
          </div>
          <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>~4.01 hours (54.4% of corpus)</p>
        </div>

        <div className="metric-box">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '13px', color: 'var(--text-secondary)', fontWeight: 600 }}>Male Speakers</span>
            <UserCheck size={18} color="var(--accent-amber)" />
          </div>
          <div className="metric-val" style={{ background: 'linear-gradient(135deg, #fde68a, #f59e0b)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            1,956
          </div>
          <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>~3.07 hours (45.6% of corpus)</p>
        </div>
      </div>

      {/* Interactive Evaluation Table */}
      <div className="glass-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '18px', flexWrap: 'wrap', gap: '12px' }}>
          <div>
            <h3 style={{ fontSize: '18px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Database size={20} color="var(--accent-cyan)" />
              10-Minute Baseline Experiment Results
            </h3>
            <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '2px' }}>
              OpenSLR-65 Tamil Test Split Utterances & Model Output Transcriptions
            </p>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                background: 'rgba(15, 23, 42, 0.8)',
                padding: '8px 14px',
                borderRadius: 'var(--radius-full)',
                border: '1px solid var(--border-subtle)',
              }}
            >
              <Search size={16} color="var(--text-muted)" />
              <input
                type="text"
                placeholder="Search Tamil text..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                style={{
                  background: 'transparent',
                  border: 'none',
                  outline: 'none',
                  color: 'white',
                  fontSize: '13px',
                  width: '160px',
                }}
              />
            </div>

            <button
              onClick={exportCSV}
              className="btn-primary"
              style={{ padding: '8px 16px', fontSize: '13px' }}
            >
              <Download size={14} /> Export CSV
            </button>
          </div>
        </div>

        <div className="data-table-container">
          <table className="custom-table">
            <thead>
              <tr>
                <th style={{ width: '80px' }}>Sample</th>
                <th>Ground Truth Reference (Tamil)</th>
                <th>Whisper-Tiny Prediction</th>
                <th>Wav2Vec2 Tamil Prediction</th>
                <th style={{ width: '120px' }}>Whisper WER / CER</th>
                <th style={{ width: '120px' }}>Wav2Vec2 WER / CER</th>
              </tr>
            </thead>
            <tbody>
              {filteredSamples.map((item, idx) => (
                <tr key={idx}>
                  <td>
                    <span className="badge-tag" style={{ fontSize: '11px' }}>#{idx + 1}</span>
                  </td>
                  <td className="tamil-text" style={{ fontWeight: 600, color: '#f8fafc', fontSize: '15px' }}>
                    {item.reference}
                  </td>
                  <td className="tamil-text" style={{ color: '#c7d2fe', fontSize: '15px' }}>
                    {item.whisper}
                  </td>
                  <td className="tamil-text" style={{ color: '#e9d5ff', fontSize: '15px' }}>
                    {item.wav2vec}
                  </td>
                  <td>
                    <span className="badge-tag badge-success" style={{ fontSize: '11px' }}>
                      {(item.wer_whisper * 100).toFixed(0)}% / {(item.cer_whisper * 100).toFixed(0)}%
                    </span>
                  </td>
                  <td>
                    <span className="badge-tag" style={{ fontSize: '11px', borderColor: 'rgba(168, 85, 247, 0.3)' }}>
                      {(item.wer_wav2vec * 100).toFixed(0)}% / {(item.cer_wav2vec * 100).toFixed(0)}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
