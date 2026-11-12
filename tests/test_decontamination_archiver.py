from lm_eval.decontamination import archiver


def test_zstd_reader_passes_filename_as_subprocess_argument(monkeypatch, tmp_path):
    compressed_file = tmp_path / "ngrams; touch injected.sorted.zst"
    compressed_file.touch()
    calls = []

    def run(args, *, check):
        calls.append((args, check))

    class FakeTextReader:
        def __init__(self, file_path):
            self.file_path = file_path

        def read_tqdm(self):
            yield "ngram 1"

    monkeypatch.setattr(archiver.subprocess, "run", run)
    monkeypatch.setattr(archiver, "TextReader", FakeTextReader)
    monkeypatch.setattr(archiver.os, "remove", lambda path: None)

    result = list(archiver.ZStdTextReader(str(compressed_file)).read_tqdm())

    assert result == ["ngram 1"]
    # Verify that the full filename remains one subprocess argument, including
    # its shell-like characters.
    assert calls == [(["zstd", "-d", "--", str(compressed_file)], True)]
