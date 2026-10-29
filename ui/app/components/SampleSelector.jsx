'use client';

import React from 'react';
import { Sparkles, Play, CheckCircle2, User, Clock } from 'lucide-react';

export default function SampleSelector({ samples, selectedSample, onSelectSample, isTranscribing }) {
  return (
    <div className="glass-card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <h3 style={{ fontSize: '18px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Sparkles size={20} color="var(--accent-cyan)" />
          OpenSLR-65 Tamil Presets
        </h3>
        <span className="badge-tag">Benchmark Verified</span>
      </div>

      <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '14px' }}>
        Select a real Tamil speech sample to test transcription and ground-truth WER/CER evaluation:
      </p>

      <div className="sample-list">
        {samples?.map((sample) => {
          const isSelected = selectedSample?.id === sample.id;
          return (
            <div
              key={sample.id}
              className={`sample-item ${isSelected ? 'selected' : ''}`}
              onClick={() => !isTranscribing && onSelectSample(sample)}
            >
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px', flex: 1 }}>
                <div
                  style={{
                    width: '32px',
                    height: '32px',
                    borderRadius: '8px',
                    background: isSelected ? 'var(--accent-indigo)' : 'rgba(255, 255, 255, 0.05)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    marginTop: '2px',
                    color: isSelected ? 'white' : 'var(--text-muted)',
                  }}
                >
                  {isSelected ? <CheckCircle2 size={16} /> : <Play size={14} />}
                </div>

                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                    <span style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)' }}>
                      {sample.title}
                    </span>
                    <span style={{ fontSize: '11px', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '3px' }}>
                      <User size={10} /> {sample.speaker}
                    </span>
                    <span style={{ fontSize: '11px', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '3px' }}>
                      <Clock size={10} /> {sample.duration_sec}s
                    </span>
                  </div>

                  <p className="tamil-text" style={{ fontSize: '13px', color: '#cbd5e1', marginTop: '4px' }}>
                    "{sample.reference}"
                  </p>

                  <p style={{ fontSize: '11px', color: 'var(--text-muted)', fontStyle: 'italic', marginTop: '2px' }}>
                    {sample.english_translation}
                  </p>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
