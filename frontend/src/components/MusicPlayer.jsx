import { useEffect, useRef } from "react";

export default function MusicPlayer({ track }) {
  const audioRef = useRef(null);

  useEffect(() => {
    if (!track?.preview_url || !audioRef.current) return;
    const playPromise = audioRef.current.play();
    if (playPromise && typeof playPromise.catch === "function") {
      playPromise.catch(() => {
      });
    }
  }, [track?.id, track?.preview_url]);

  if (!track) {
    return (
      <div className="glass p-5 animate-floatIn">
        <h3 className="font-heading text-lg text-text0">Music Player</h3>
        <p className="mt-3 text-sm text-text1">No track selected yet. Start detection to load music.</p>
      </div>
    );
  }

  return (
    <div className="glass p-5 animate-floatIn">
      <h3 className="font-heading text-lg text-text0">Music Player</h3>
      <p className="mt-2 text-sm text-text0">
        {track.name} <span className="text-text1">- {track.artist}</span>
      </p>

      {track.preview_url ? (
        <audio
          key={track.id}
          ref={audioRef}
          className="mt-4 w-full"
          src={track.preview_url}
          controls
          autoPlay
          preload="auto"
        />
      ) : null}

      {!track.preview_url && track.embed_url ? (
        <iframe
          title={`spotify-${track.id}`}
          className="mt-4 h-40 w-full rounded-xl border border-white/10"
          src={track.embed_url}
          allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
          loading="lazy"
        />
      ) : null}

      {!track.preview_url && !track.embed_url ? (
        <div className="mt-4">
          <p className="text-xs text-text1">In-app preview is unavailable for this track. Open externally to play full song.</p>
          <a
            href={track.external_url || "#"}
            target="_blank"
            rel="noreferrer"
            className="mt-2 inline-flex text-sm text-accent hover:underline"
          >
            Open in Spotify
          </a>
        </div>
      ) : null}
    </div>
  );
}
