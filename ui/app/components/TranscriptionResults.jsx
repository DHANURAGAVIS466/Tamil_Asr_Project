'use client';

import React, { useState } from 'react';
import { Bot, Zap, Copy, Check, BarChart2, ShieldCheck, HelpCircle } from 'lucide-react';

export default function TranscriptionResults({ results, isTranscribing }) {
  const [copiedWhisper, setCopiedWhisper] = useState(false);
  const [copiedWav2vec, setCopiedWav2vec] = useState(false);

  if (isTranscribing) {
    return (
      <div className="glass-card" style={{ textAlign: 'center', padding: '60px 20px' }}>
        <div
          style={{
            width: '60px',
            height: '60px',
            borderRadius: '50%',
            border: '4px solid rgba(99, 102, 241, 0.2)',
            borderTopColor: 'var(--accent-indigo)',
            animation: 'spin 1s linear infinite',
            margin: '0 auto 20px',
          }}
        />
        <h3 style={{ fontSize: '20px', marginBottom: '8px' }}>Processing Tamil Speech Recognition</h3>
        <p style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>
          Resampling to 16 kHz & running Whisper-Tiny & Wav2Vec2 inference...
        </p>
        <style jsx>{`
          @keyframes spin {
            to {
              transform: rotate(360deg);
            }
          }
        `}</style>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="glass-card" style={{ textAlign: 'center', padding: '60px 20px', color: 'var(--text-muted)' }}>
        <Bot size={48} style={{ margin: '0 auto 16px', opacity: 0.4 }} />
        <h3 style={{ fontSize: '18px', color: 'var(--text-secondary)', marginBottom: '6px' }}>
          Awaiting Tamil Audio Input
        </h3>
        <p style={{ fontSize: '14px' }}>
          Record audio from your microphone, upload a file, or pick an OpenSLR-65 preset on the left.
        </p>
      </div>
    );
  }

  const whisperData = results?.predictions?.whisper;
  const wav2vecData = results?.predictions?.wav2vec2;
  const reference = results?.reference;

  const copyText = (text, type) => {
    navigator.clipboard.writeText(text);
    if (type === 'whisper') {
      setCopiedWhisper(true);
      setTimeout(() => setCopiedWhisper(false), 2000);
    } else {
      setCopiedWav2vec(true);
      setTimeout(() => setCopiedWav2vec(false), 2000);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Ground Truth Reference Banner (if provided) */}
      {reference && (
        <div
          className="glass-card"
          style={{
            background: 'rgba(6, 182, 212, 0.08)',
            border: '1px solid rgba(6, 182, 212, 0.25)',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <span style={{ fontSize: '12px', fontWeight: 700, color: 'var(--accent-cyan)', textTransform: 'uppercase' }}>
              Ground Truth Reference (OpenSLR-65)
            </span>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
              Duration: {results.audio_duration_sec}s
            </span>
          </div>
          <p className="tamil-text" style={{ fontSize: '18px', color: '#f1f5f9', fontWeight: 600 }}>
            {reference}
          </p>
        </div>
      )}

      {/* Whisper-Tiny Output Card */}
      <div className="model-output-card">
        <div className="model-card-header">
          <div className="model-title-box">
            <div
              style={{
                width: '36px',
                height: '36px',
                borderRadius: '8px',
                background: 'rgba(99, 102, 241, 0.2)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'var(--accent-indigo)',
              }}
            >
              <Bot size={20} />
            </div>
            <div>
              <h4 style={{ fontSize: '16px', fontWeight: 700 }}>Whisper-Tiny (39M)</h4>
              <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                Multilingual Sequence-to-Sequence • CPU Ready
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            {whisperData?.latency_sec > 0 && (
              <span className="badge-tag" style={{ fontSize: '11px' }}>
                <Zap size={12} /> {whisperData.latency_sec}s ({whisperData.rtf}x RTF)
              </span>
            )}
            <button
              onClick={() => copyText(whisperData?.text, 'whisper')}
              style={{
                background: 'rgba(255, 255, 255, 0.06)',
                border: 'none',
                borderRadius: '6px',
                color: 'var(--text-secondary)',
                padding: '6px 10px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                fontSize: '12px',
              }}
            >
              {copiedWhisper ? <Check size={14} color="#10b981" /> : <Copy size={14} />}
              {copiedWhisper ? 'Copied' : 'Copy'}
            </button>
          </div>
        </div>

        <div className="transcription-box tamil-text">
          {whisperData?.text || 'No transcription generated'}
        </div>

        {/* Diff Highlighting if Ground Truth available */}
        {whisperData?.metrics?.diffs && (
          <div>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>
              Token Error Breakdown (WER: {whisperData.metrics.wer_percent} | CER: {whisperData.metrics.cer_percent}):
            </span>
            <div className="diff-container">
              {whisperData.metrics.diffs.map((diff, idx) => (
                <span
                  key={idx}
                  className={`diff-token ${
                    diff.type === 'match'
                      ? 'diff-match'
                      : diff.type === 'substitution'
                      ? 'diff-sub'
                      : diff.type === 'deletion'
                      ? 'diff-del'
                      : 'diff-ins'
                  }`}
                  title={`${diff.type}: Ref="${diff.ref}", Hyp="${diff.hyp}"`}
                >
                  {diff.hyp || `[del: ${diff.ref}]`}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Wav2Vec2 Output Card */}
      <div className="model-output-card wav2vec">
        <div className="model-card-header">
          <div className="model-title-box">
            <div
              style={{
                width: '36px',
                height: '36px',
                borderRadius: '8px',
                background: 'rgba(168, 85, 247, 0.2)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'var(--accent-purple)',
              }}
            >
              <ShieldCheck size={20} />
            </div>
            <div>
              <h4 style={{ fontSize: '16px', fontWeight: 700 }}>Wav2Vec2-Large-XLSR-53 (Tamil)</h4>
              <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                Fine-tuned CTC Acoustic Model (315M)
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            {wav2vecData?.latency_sec > 0 && (
              <span className="badge-tag" style={{ fontSize: '11px' }}>
                <Zap size={12} /> {wav2vecData.latency_sec}s
              </span>
            )}
            <button
              onClick={() => copyText(wav2vecData?.text, 'wav2vec')}
              style={{
                background: 'rgba(255, 255, 255, 0.06)',
                border: 'none',
                borderRadius: '6px',
                color: 'var(--text-secondary)',
                padding: '6px 10px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                fontSize: '12px',
              }}
            >
              {copiedWav2vec ? <Check size={14} color="#10b981" /> : <Copy size={14} />}
              {copiedWav2vec ? 'Copied' : 'Copy'}
            </button>
          </div>
        </div>

        <div className="transcription-box tamil-text">
          {wav2vecData?.text || 'No transcription generated'}
        </div>

        {/* Wav2Vec2 Diff */}
        {wav2vecData?.metrics?.diffs && (
          <div>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>
              Token Error Breakdown (WER: {wav2vecData.metrics.wer_percent} | CER: {wav2vecData.metrics.cer_percent}):
            </span>
            <div className="diff-container">
              {wav2vecData.metrics.diffs.map((diff, idx) => (
                <span
                  key={idx}
                  className={`diff-token ${
                    diff.type === 'match'
                      ? 'diff-match'
                      : diff.type === 'substitution'
                      ? 'diff-sub'
                      : diff.type === 'deletion'
                      ? 'diff-del'
                      : 'diff-ins'
                  }`}
                  title={`${diff.type}: Ref="${diff.ref}", Hyp="${diff.hyp}"`}
                >
                  {diff.hyp || `[del: ${diff.ref}]`}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
