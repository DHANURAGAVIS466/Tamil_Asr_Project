import './globals.css';

export const metadata = {
  title: 'Tamil ASR Studio | Whisper & Wav2Vec2 Benchmarking',
  description: 'High-performance Tamil Automatic Speech Recognition studio and evaluation benchmark dashboard powered by Whisper-Tiny and Wav2Vec2.',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Mukta+Malar:wght@400;500;600;700;800&family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        <div className="app-container">
          {children}
        </div>
      </body>
    </html>
  );
}
