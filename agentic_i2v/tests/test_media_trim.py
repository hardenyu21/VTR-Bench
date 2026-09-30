from __future__ import annotations

import unittest

import numpy as np

from agentic_i2v.media.first_frame_trim import trim_conditioning_frame, trim_leading_audio_frame


class MediaTrimTests(unittest.TestCase):
    def test_drop_conditioning_frame(self) -> None:
        frames = np.arange(241)
        result = trim_conditioning_frame(frames)
        self.assertEqual(len(result), 240)
        self.assertEqual(result[0], 1)
        self.assertEqual(result[-1], 240)

    def test_drop_conditioning_frame_for_five_second_smoke(self) -> None:
        frames = np.arange(124)
        result = trim_conditioning_frame(frames, requested=121, delivered=120)
        self.assertEqual(len(result), 120)
        self.assertEqual(result[0], 1)
        self.assertEqual(result[-1], 120)

    def test_audio_is_shifted_one_frame_and_ten_seconds(self) -> None:
        audio = np.arange(24100)
        result = trim_leading_audio_frame(audio, audio_rate=2400)
        self.assertEqual(len(result), 24000)
        self.assertEqual(result[0], 100)


if __name__ == "__main__":
    unittest.main()
