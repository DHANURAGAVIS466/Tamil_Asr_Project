'use client';

import React from 'react';
import { Mic, Cpu, Sparkles, Activity, ShieldCheck } from 'lucide-react';

export default function Header({ serverStatus }) {
  const isHealthy = serverStatus?.status === 'healthy';
  const whisperLoaded = serverStatus?.models?.whisper?.loaded;
  const wav2vecLoaded = serverStatus?.models?.wav2vec2?.loaded;

  return (
    <header className="header-container">
      <div className="brand-section">
        <div className="logo-icon-wrapper">
          <Mic size={28} color="#ffffff" />
        </div>
        <div>
          <h1 className="brand-title">
            தமிழ் ASR Studio
            <span className="badge-tag">
              <Sparkles size={13} /> OpenSLR-65
            </span>
          </h1>
          <p className="brand-subtitle">
            Tamil Speech Recognition & Evaluation Benchmark • Whisper & Wav2Vec2
          </p>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
        <div className={`badge-tag ${isHealthy ? 'badge-success' : 'badge-warning'}`}>
          <Activity size={14} />
          {isHealthy ? `Backend: ${serverStatus.device || 'Online'}` : 'Connecting Engine...'}
        </div>

        <div className="badge-tag">
          <Cpu size={14} />
          Whisper-Tiny (39M) {whisperLoaded ? '✓' : '...'}
        </div>

        <div className="badge-tag">
          <ShieldCheck size={14} />
          Wav2Vec2 Tamil {wav2vecLoaded ? '✓' : '(Shadow)'}
        </div>
      </div>
    </header>
  );
}
