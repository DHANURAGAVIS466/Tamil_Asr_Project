'use client';

import React from 'react';
import { Layers, CheckCircle, ArrowRight, Zap, BookOpen, ShieldCheck, FileCheck, Sparkles } from 'lucide-react';

export default function RoadmapViewer() {
  const phases = [
    {
      step: 1,
      title: 'Pretrained Whisper Baseline',
      desc: 'Evaluate Whisper-Tiny on OpenSLR-65 Tamil without fine-tuning to establish zero-shot baseline WER/CER on CPU.',
      status: 'completed',
      badge: 'Completed (Current)'
    },
    {
      step: 2,
      title: 'Tamil Wav2Vec2 Baseline',
      desc: 'Benchmark Amrrs/wav2vec2-large-xlsr-53-tamil CTC acoustic model on identical 16-kHz speech segments.',
      status: 'completed',
      badge: 'Completed (Current)'
    },
    {
      step: 3,
      title: 'Domain Adaptation & Fine-Tuning',
      desc: 'Fine-tune acoustic and language models with LoRA/PEFT on target Tamil domain vocabularies (e.g., administrative records).',
      status: 'current',
      badge: 'Next Milestone'
    },
    {
      step: 4,
      title: 'Multi-Model Fusion (Whisper + Wav2Vec2)',
      desc: 'Combine acoustic CTC alignments with Seq2Seq autoregressive decoding via Minimum Bayes Risk (MBR) / ROVER fusion.',
      status: 'future',
      badge: 'Research Phase'
    },
    {
      step: 5,
      title: 'Government Terminology Validation',
      desc: 'Integrate custom Tamil administrative entity dictionaries and phoneme validation rules to prevent domain hallucinations.',
      status: 'future',
      badge: 'Domain Integration'
    },
    {
      step: 6,
      title: 'Context-Aware Error Correction',
      desc: 'Leverage Tamil LLM post-editing to correct complex agglutinative morphology and punctuation errors.',
      status: 'future',
      badge: 'NLP Post-Processing'
    },
    {
      step: 7,
      title: 'Structured Government Document Output',
      desc: 'Transform raw transcribed speech into structured official Tamil GO/Gazette templates with metadata tagging.',
      status: 'future',
      badge: 'Final Proposed System'
    }
  ];

  return (
    <div className="glass-card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <div>
          <h3 style={{ fontSize: '20px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Layers size={22} color="var(--accent-indigo)" />
            Government Tamil ASR Research Roadmap
          </h3>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '4px' }}>
            Systematic 7-Phase Architecture: From Baseline Pretrained Inference to Structured Document Synthesis
          </p>
        </div>
        <span className="badge-tag">
          <Sparkles size={13} /> Project Pipeline
        </span>
      </div>

      <div style={{ marginTop: '16px' }}>
        {phases.map((p) => (
          <div key={p.step} className="pipeline-step">
            <div className={`step-indicator ${p.status === 'completed' || p.status === 'current' ? 'step-active' : 'step-future'}`}>
              {p.status === 'completed' ? <CheckCircle size={14} /> : p.step}
            </div>

            <div
              style={{
                background: p.status === 'current' ? 'rgba(99, 102, 241, 0.12)' : 'rgba(15, 23, 42, 0.5)',
                border: p.status === 'current' ? '1px solid var(--border-highlight)' : '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-md)',
                padding: '16px 20px',
                transition: 'all 0.2s ease',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px', flexWrap: 'wrap', gap: '8px' }}>
                <h4 style={{ fontSize: '16px', fontWeight: 700, color: 'var(--text-primary)' }}>
                  Phase {p.step}: {p.title}
                </h4>
                <span
                  className={`badge-tag ${
                    p.status === 'completed'
                      ? 'badge-success'
                      : p.status === 'current'
                      ? ''
                      : 'badge-warning'
                  }`}
                  style={{ fontSize: '11px' }}
                >
                  {p.badge}
                </span>
              </div>

              <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                {p.desc}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
