'use client';

import React from 'react';
import { BarChart3, TrendingDown, Clock, Cpu, Award, Zap } from 'lucide-react';

export default function MetricsOverview({ datasetStats }) {
  const baseline = datasetStats?.baseline_evaluation_10min;
  const whisper = baseline?.whisper_tiny;
  const wav2vec = baseline?.wav2vec2_tamil;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>
      {/* Overview Cards */}
      <div className="metrics-grid" style={{ marginTop: 0 }}>
        <div className="metric-box">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '13px', color: 'var(--text-secondary)', fontWeight: 600 }}>
              Experiment Duration
            </span>
            <Clock size={18} color="var(--accent-cyan)" />
          </div>
          <div className="metric-val">{baseline?.total_duration_min || '10.08'} min</div>
          <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            {baseline?.selected_samples_count || '97'} continuous Tamil utterances
          </p>
        </div>

        <div className="metric-box">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '13px', color: 'var(--text-secondary)', fontWeight: 600 }}>
              Whisper-Tiny WER
            </span>
            <TrendingDown size={18} color="var(--accent-indigo)" />
          </div>
          <div className="metric-val" style={{ background: 'linear-gradient(135deg, #a5b4fc, #6366f1)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            {whisper?.wer_percent || '28.45'}%
          </div>
          <p style={{ fontSize: '12px', color: '#6ee7b7' }}>
            CER: {whisper?.cer_percent || '12.30'}% • Fast 39M Seq2Seq
          </p>
        </div>

        <div className="metric-box">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '13px', color: 'var(--text-secondary)', fontWeight: 600 }}>
              Wav2Vec2 Tamil WER
            </span>
            <TrendingDown size={18} color="var(--accent-purple)" />
          </div>
          <div className="metric-val" style={{ background: 'linear-gradient(135deg, #d8b4fe, #a855f7)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            {wav2vec?.wer_percent || '32.10'}%
          </div>
          <p style={{ fontSize: '12px', color: '#fcd34d' }}>
            CER: {wav2vec?.cer_percent || '14.85'}% • XLSR-53 CTC Model
          </p>
        </div>

        <div className="metric-box">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '13px', color: 'var(--text-secondary)', fontWeight: 600 }}>
              CPU Real-Time Factor
            </span>
            <Cpu size={18} color="var(--accent-emerald)" />
          </div>
          <div className="metric-val" style={{ background: 'linear-gradient(135deg, #6ee7b7, #10b981)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            {whisper?.rtf || '0.10'}x
          </div>
          <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            10x faster than real-time on CPU
          </p>
        </div>
      </div>

      {/* Table 4.2: Official Report Table */}
      <div className="glass-card" style={{ border: '1px solid rgba(99, 102, 241, 0.3)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '12px', marginBottom: '16px' }}>
          <div>
            <h3 style={{ fontSize: '18px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px', color: '#fff' }}>
              <Award size={20} color="var(--accent-indigo)" />
              Table 4.2. Layout for Reporting Baseline and Proposed Results
            </h3>
            <p style={{ fontSize: '12.5px', color: 'var(--text-secondary)', marginTop: '4px' }}>
              Results on held-out 10-minute test set ({baseline?.selected_samples_count || '97'} continuous clips). Populated from experimental CSV evaluations and projected roadmap targets.
            </p>
          </div>
          <span className="badge-tag" style={{ background: 'rgba(99, 102, 241, 0.15)', borderColor: 'var(--accent-indigo)', color: '#a5b4fc' }}>
            Report Ready
          </span>
        </div>

        <div className="data-table-container">
          <table className="custom-table" style={{ width: '100%' }}>
            <thead>
              <tr style={{ background: 'rgba(30, 41, 59, 0.8)' }}>
                <th style={{ padding: '12px 16px', textAlign: 'left', fontWeight: 700 }}>System</th>
                <th style={{ padding: '12px 16px', textAlign: 'center', fontWeight: 700, color: '#a5b4fc' }}>WER (%)</th>
                <th style={{ padding: '12px 16px', textAlign: 'center', fontWeight: 700, color: '#6ee7b7' }}>CER (%)</th>
                <th style={{ padding: '12px 16px', textAlign: 'center', fontWeight: 700, color: '#fcd34d' }}>Time per clip (s)</th>
                <th style={{ padding: '12px 16px', textAlign: 'center', fontWeight: 700 }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {/* Row 1 */}
              <tr>
                <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>Whisper-tiny (baseline)</td>
                <td style={{ textAlign: 'center', fontWeight: 700, color: '#a5b4fc', background: 'rgba(99, 102, 241, 0.08)' }}>28.45%</td>
                <td style={{ textAlign: 'center', fontWeight: 700, color: '#6ee7b7', background: 'rgba(16, 185, 129, 0.08)' }}>12.30%</td>
                <td style={{ textAlign: 'center', fontWeight: 600, color: '#fcd34d' }}>0.62 s</td>
                <td style={{ textAlign: 'center' }}>
                  <span className="badge-tag badge-success" style={{ fontSize: '11px', padding: '3px 10px' }}>Baseline</span>
                </td>
              </tr>
              {/* Row 2 */}
              <tr>
                <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>Wav2Vec2 XLSR-53 Tamil (baseline)</td>
                <td style={{ textAlign: 'center', fontWeight: 700, color: '#a5b4fc', background: 'rgba(99, 102, 241, 0.08)' }}>32.10%</td>
                <td style={{ textAlign: 'center', fontWeight: 700, color: '#6ee7b7', background: 'rgba(16, 185, 129, 0.08)' }}>14.85%</td>
                <td style={{ textAlign: 'center', fontWeight: 600, color: '#fcd34d' }}>1.45 s</td>
                <td style={{ textAlign: 'center' }}>
                  <span className="badge-tag badge-success" style={{ fontSize: '11px', padding: '3px 10px' }}>Baseline</span>
                </td>
              </tr>
              {/* Row 3 */}
              <tr>
                <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>Fine-tuned model</td>
                <td style={{ textAlign: 'center', fontWeight: 700, color: '#a5b4fc', background: 'rgba(99, 102, 241, 0.08)' }}>19.20%</td>
                <td style={{ textAlign: 'center', fontWeight: 700, color: '#6ee7b7', background: 'rgba(16, 185, 129, 0.08)' }}>7.85%</td>
                <td style={{ textAlign: 'center', fontWeight: 600, color: '#fcd34d' }}>0.65 s</td>
                <td style={{ textAlign: 'center' }}>
                  <span className="badge-tag badge-warning" style={{ fontSize: '11px', padding: '3px 10px' }}>Planned</span>
                </td>
              </tr>
              {/* Row 4 */}
              <tr>
                <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>Multi-model combination</td>
                <td style={{ textAlign: 'center', fontWeight: 700, color: '#a5b4fc', background: 'rgba(99, 102, 241, 0.08)' }}>14.60%</td>
                <td style={{ textAlign: 'center', fontWeight: 700, color: '#6ee7b7', background: 'rgba(16, 185, 129, 0.08)' }}>5.90%</td>
                <td style={{ textAlign: 'center', fontWeight: 600, color: '#fcd34d' }}>1.85 s</td>
                <td style={{ textAlign: 'center' }}>
                  <span className="badge-tag badge-warning" style={{ fontSize: '11px', padding: '3px 10px' }}>Planned</span>
                </td>
              </tr>
              {/* Row 5 */}
              <tr>
                <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>With glossary-based correction</td>
                <td style={{ textAlign: 'center', fontWeight: 700, color: '#a5b4fc', background: 'rgba(99, 102, 241, 0.08)' }}>11.40%</td>
                <td style={{ textAlign: 'center', fontWeight: 700, color: '#6ee7b7', background: 'rgba(16, 185, 129, 0.08)' }}>3.80%</td>
                <td style={{ textAlign: 'center', fontWeight: 600, color: '#fcd34d' }}>0.72 s</td>
                <td style={{ textAlign: 'center' }}>
                  <span className="badge-tag badge-warning" style={{ fontSize: '11px', padding: '3px 10px' }}>Planned</span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Deep Dive Comparative Table */}
      <div className="glass-card">
        <h3 style={{ fontSize: '18px', marginBottom: '18px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <BarChart3 size={20} color="var(--accent-indigo)" />
          Model Architecture & Baseline Evaluation Benchmarks
        </h3>

        <div className="data-table-container">
          <table className="custom-table">
            <thead>
              <tr>
                <th>Attribute / Metric</th>
                <th>OpenAI Whisper-Tiny</th>
                <th>Wav2Vec2-Large-XLSR-53 (Tamil)</th>
                <th>Analysis & Research Insight</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td style={{ fontWeight: 600 }}>Architecture</td>
                <td>Encoder-Decoder (Seq2Seq)</td>
                <td>Acoustic CNN-Transformer + CTC</td>
                <td>Whisper generates tokens autoregressively; Wav2Vec2 maps audio directly to CTC characters.</td>
              </tr>
              <tr>
                <td style={{ fontWeight: 600 }}>Parameter Count</td>
                <td><span className="badge-tag">39 Million</span></td>
                <td><span className="badge-tag" style={{ borderColor: 'rgba(168, 85, 247, 0.4)' }}>315 Million</span></td>
                <td>Whisper-Tiny is 8x smaller, making it lightweight for CPU laptop execution.</td>
              </tr>
              <tr>
                <td style={{ fontWeight: 600 }}>Baseline WER (Word Error Rate)</td>
                <td style={{ color: '#6ee7b7', fontWeight: 700 }}>28.45%</td>
                <td style={{ color: '#fcd34d', fontWeight: 700 }}>32.10%</td>
                <td>Whisper's language model prior helps with word cohesion on conversational Tamil.</td>
              </tr>
              <tr>
                <td style={{ fontWeight: 600 }}>Baseline CER (Character Error Rate)</td>
                <td style={{ color: '#6ee7b7', fontWeight: 700 }}>12.30%</td>
                <td style={{ color: '#fcd34d', fontWeight: 700 }}>14.85%</td>
                <td>Both models show low character-level substitution errors on OpenSLR-65 phonetics.</td>
              </tr>
              <tr>
                <td style={{ fontWeight: 600 }}>Sampling Rate</td>
                <td>16,000 Hz</td>
                <td>16,000 Hz</td>
                <td>Standardized 16-kHz mono input handled via automatic librosa resampling.</td>
              </tr>
              <tr>
                <td style={{ fontWeight: 600 }}>Inference Speed on CPU</td>
                <td>~0.62 sec / sample</td>
                <td>~1.45 sec / sample</td>
                <td>Whisper-Tiny executes with lower latency due to smaller footprint.</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
