# Sprint 0 Artifact Inventory

Artifacts are stored in Google Drive, not GitHub.

## Google Drive Base

```text
/content/drive/MyDrive/parakeet_bilingual_asr/artifacts/sprint0/
```

## Sprint 0 Artifacts Produced

```text
step1_environment_snapshot.json
step2_nemo_environment_snapshot.json
steps1_2_environment_snapshot.json
step3_smoke_test/
  smoke_test_result.json
  synthetic_silence_16k.wav
step4_audio_validation/
  seerah_01_audio_metadata.json
  seerah_01_normalized_16k_mono.wav
step5_asr_test_clip/
  seerah_01_test_clip_60s.wav
step7_parakeet_api/
  parakeet_hosted_api_smoke_test_raw.json
step7_5_parakeet_metadata/
  parakeet_metadata_capability_raw.json
  parakeet_metadata_capability_raw.txt
step8_parakeet_exports/
  seerah_01_test_clip_parakeet_clean.json
  seerah_01_test_clip_parakeet_clean.txt
```

## Notes

- The normalized full WAV is large and should remain in Drive.
- The 60-second clip is useful for future smoke tests.
- The Step 7.5 raw output contains repeated partial streaming hypotheses.
- The Step 8 clean JSON/TXT contains final transcript blocks and word timestamps.
