import os


def main():
    try:
        # Prefer macOS cache path; Bark will use its default if not set
        cache_dir = os.path.expanduser("~/Library/Caches/suno/bark")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ.setdefault("BARK_CACHE_DIR", cache_dir)

        from bark import preload_models

        preload_models()
        print("✅ All Bark models and presets are cached locally.")
        print(f"📍 Cache directory: {cache_dir}")
    except Exception as e:
        print(f"❌ Failed to preload Bark assets: {e}")


if __name__ == "__main__":
    main()


