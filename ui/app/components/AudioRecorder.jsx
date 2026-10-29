'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Mic, Square, Play, Pause, RotateCcw, Sparkles } from 'lucide-react';

export default function AudioRecorder({ onAudioReady, isTranscribing }) {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerRef = useRef(null);
  const canvasRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const animationFrameRef = useRef(null);
  const audioElementRef = useRef(null);

  // Setup Canvas Wave Animation
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const draw = () => {
      animationFrameRef.current = requestAnimationFrame(draw);
      const width = canvas.width;
      const height = canvas.height;

      ctx.clearRect(0, 0, width, height);

      if (isRecording && analyserRef.current) {
        const bufferLength = analyserRef.current.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        analyserRef.current.getByteFrequencyData(dataArray);

        const barWidth = (width / bufferLength) * 2.5;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
          const barHeight = (dataArray[i] / 255) * (height * 0.85);
          const gradient = ctx.createLinearGradient(0, height, 0, 0);
          gradient.addColorStop(0, '#6366f1');
          gradient.addColorStop(0.5, '#a855f7');
          gradient.addColorStop(1, '#06b6d4');

          ctx.fillStyle = gradient;
          ctx.beginPath();
          ctx.roundRect(x, (height - barHeight) / 2, barWidth - 1, barHeight, [3]);
          ctx.fill();

          x += barWidth + 1;
        }
      } else {
        // Idle ambient subtle wave
        const time = Date.now() * 0.002;
        ctx.beginPath();
        ctx.moveTo(0, height / 2);
        for (let x = 0; x < width; x += 4) {
          const y = height / 2 + Math.sin(x * 0.03 + time) * 6;
          ctx.lineTo(x, y);
        }
        ctx.strokeStyle = 'rgba(99, 102, 241, 0.3)';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    };

    draw();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isRecording]);

  const [micError, setMicError] = useState(null);

  const startRecording = async () => {
    try {
      setMicError(null);
      audioChunksRef.current = [];
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // Audio Context for live waveform
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 64;
      const source = audioCtx.createMediaStreamSource(stream);
      source.connect(analyser);

      audioContextRef.current = audioCtx;
      analyserRef.current = analyser;

      // Pick browser-supported audio mimeType
      const candidateTypes = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/ogg;codecs=opus',
        'audio/mp4',
      ];
      let mimeType = '';
      if (typeof MediaRecorder.isTypeSupported === 'function') {
        for (const type of candidateTypes) {
          if (MediaRecorder.isTypeSupported(type)) {
            mimeType = type;
            break;
          }
        }
      }

      const mediaRecorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      const finalMime = mediaRecorder.mimeType || mimeType || 'audio/webm';

      mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const ext = finalMime.includes('webm')
          ? 'webm'
          : finalMime.includes('ogg')
          ? 'ogg'
          : finalMime.includes('mp4')
          ? 'mp4'
          : 'wav';
        const blob = new Blob(audioChunksRef.current, { type: finalMime });
        const url = URL.createObjectURL(blob);
        setAudioBlob(blob);
        setAudioUrl(url);
        if (onAudioReady) {
          onAudioReady(blob, `microphone_recording.${ext}`);
        }
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setRecordingTime(0);

      timerRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      console.error('Microphone access denied or unavailable:', err);
      setMicError(
        'Microphone access is blocked or unavailable. Please click the icon in your address bar (🔒 / 🎙️) and set Microphone to "Allow".'
      );
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      clearInterval(timerRef.current);
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    }
  };

  const resetRecording = () => {
    setAudioBlob(null);
    setAudioUrl(null);
    setRecordingTime(0);
    setIsPlaying(false);
  };

  const togglePlayback = () => {
    if (!audioElementRef.current) return;
    if (isPlaying) {
      audioElementRef.current.pause();
      setIsPlaying(false);
    } else {
      audioElementRef.current.play();
      setIsPlaying(true);
    }
  };

  const formatTime = (secs) => {
    const mins = Math.floor(secs / 60);
    const remainingSecs = secs % 60;
    return `${mins.toString().padStart(2, '0')}:${remainingSecs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="glass-card recorder-card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 style={{ fontSize: '18px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Mic size={20} color="var(--accent-indigo)" />
          Live Microphone Studio
        </h3>
        <span style={{ fontSize: '13px', color: 'var(--text-muted)' }}>16-kHz Auto Sample</span>
      </div>

      <div className="visualizer-canvas-container">
        <canvas ref={canvasRef} width={400} height={120} className="visualizer-canvas" />
        {isRecording && (
          <div
            style={{
              position: 'absolute',
              top: '12px',
              right: '12px',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              background: 'rgba(244, 63, 94, 0.2)',
              padding: '4px 10px',
              borderRadius: '20px',
              border: '1px solid rgba(244, 63, 94, 0.4)',
              color: '#fda4af',
              fontSize: '12px',
              fontWeight: 600,
            }}
          >
            <span
              style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                background: '#f43f5e',
                display: 'inline-block',
              }}
            />
            REC {formatTime(recordingTime)}
          </div>
        )}
      </div>

      <div className="record-btn-wrapper">
        {!isRecording ? (
          <button
            onClick={startRecording}
            className="btn-record"
            title="Start Recording Tamil Speech"
            disabled={isTranscribing}
          >
            <Mic size={28} />
          </button>
        ) : (
          <button
            onClick={stopRecording}
            className="btn-record recording"
            title="Stop Recording"
            disabled={isTranscribing}
          >
            <Square size={24} />
          </button>
        )}
      </div>

      <div style={{ textAlign: 'center', fontSize: '12px', color: 'var(--text-muted)' }}>
        {isRecording ? (
          <span style={{ color: '#fda4af' }}>🎙️ Recording... Speak clearly in Tamil (2–6 seconds)</span>
        ) : (
          <span>💡 Tip: Record a complete sentence (2–6 seconds) for accurate recognition</span>
        )}
      </div>

      {micError && (
        <div
          style={{
            background: 'rgba(239, 68, 68, 0.12)',
            border: '1px solid rgba(239, 68, 68, 0.35)',
            borderRadius: 'var(--radius-md)',
            padding: '12px 16px',
            fontSize: '13px',
            color: '#fca5a5',
            lineHeight: '1.5',
            display: 'flex',
            flexDirection: 'column',
            gap: '8px',
          }}
        >
          <div style={{ fontWeight: 600, color: '#f87171' }}>⚠️ Microphone Access Needed</div>
          <div>{micError}</div>
          <div style={{ fontSize: '12px', color: '#cbd5e1' }}>
            <strong>How to fix in Chrome/Edge:</strong> Click the <strong>tune/lock icon (⚙️ / 🔒)</strong> directly to the left of <code>localhost:3000</code> in your browser address bar &rarr; switch <strong>Microphone</strong> to <strong>Allow</strong> &rarr; refresh the page.
          </div>
        </div>
      )}

      {audioUrl && (
        <div
          style={{
            background: 'rgba(15, 23, 42, 0.6)',
            padding: '14px 18px',
            borderRadius: 'var(--radius-md)',
            border: '1px solid var(--border-subtle)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '12px',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <button
              onClick={togglePlayback}
              style={{
                width: '38px',
                height: '38px',
                borderRadius: '50%',
                background: 'var(--accent-indigo)',
                border: 'none',
                color: 'white',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
              }}
            >
              {isPlaying ? <Pause size={18} /> : <Play size={18} />}
            </button>
            <audio
              ref={audioElementRef}
              src={audioUrl}
              onEnded={() => setIsPlaying(false)}
              style={{ display: 'none' }}
            />
            <div>
              <p style={{ fontSize: '14px', fontWeight: 600 }}>Mic Capture ({formatTime(recordingTime)})</p>
              <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Ready for Whisper & Wav2Vec2</p>
            </div>
          </div>

          <button
            onClick={resetRecording}
            style={{
              background: 'transparent',
              border: 'none',
              color: 'var(--text-muted)',
              cursor: 'pointer',
              padding: '6px',
            }}
            title="Discard & Re-record"
          >
            <RotateCcw size={18} />
          </button>
        </div>
      )}
    </div>
  );
}
