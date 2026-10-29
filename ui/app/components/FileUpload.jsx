'use client';

import React, { useState, useRef } from 'react';
import { UploadCloud, FileAudio, Check, AlertCircle } from 'lucide-react';

export default function FileUpload({ onFileSelect, isTranscribing }) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const processFile = (file) => {
    const validTypes = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/ogg', 'audio/flac', 'audio/m4a', 'audio/x-wav'];
    if (!validTypes.includes(file.type) && !file.name.match(/\.(wav|mp3|ogg|flac|m4a)$/i)) {
      alert('Please upload a valid audio file (.wav, .mp3, .ogg, .flac).');
      return;
    }
    setSelectedFile(file);
    if (onFileSelect) {
      onFileSelect(file, file.name);
    }
  };

  return (
    <div className="glass-card">
      <h3 style={{ fontSize: '18px', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
        <UploadCloud size={20} color="var(--accent-purple)" />
        Upload Tamil Audio
      </h3>

      <div
        className={`dropzone-container ${dragActive ? 'dragging' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*,.wav,.mp3,.ogg,.flac,.m4a"
          onChange={handleChange}
          style={{ display: 'none' }}
          disabled={isTranscribing}
        />

        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px' }}>
          <div
            style={{
              width: '48px',
              height: '48px',
              borderRadius: '50%',
              background: 'rgba(168, 85, 247, 0.15)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'var(--accent-purple)',
            }}
          >
            <FileAudio size={24} />
          </div>

          <div>
            <p style={{ fontSize: '15px', fontWeight: 600, color: 'var(--text-primary)' }}>
              {selectedFile ? selectedFile.name : 'Drag & drop Tamil audio here'}
            </p>
            <p style={{ fontSize: '13px', color: 'var(--text-muted)', marginTop: '4px' }}>
              Supports WAV, MP3, FLAC, OGG (16-kHz auto converted)
            </p>
          </div>

          {selectedFile && (
            <span className="badge-tag badge-success" style={{ marginTop: '6px' }}>
              <Check size={12} /> {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB Ready
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
