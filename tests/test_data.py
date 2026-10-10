from charlm.data import DEFAULT_NAMES, build_dataset, build_vocab, load_words, split_words


def test_vocab_puts_dot_first() -> None:
    stoi, itos = build_vocab(["ab", "ba"])
    assert stoi == {".": 0, "a": 1, "b": 2}
    assert itos[2] == "b"


def test_dataset_contexts() -> None:
    stoi, _ = build_vocab(["em"])
    x, y = build_dataset(["em"], stoi, block_size=3)
    assert x.shape == (3, 3)
    assert y.tolist() == [stoi["e"], stoi["m"], 0]
    assert x[0].tolist() == [0, 0, 0]
    assert x[1].tolist() == [0, 0, stoi["e"]]
    assert x[2].tolist() == [0, stoi["e"], stoi["m"]]


def test_split_is_80_10_10_and_deterministic() -> None:
    words = [f"w{i}" for i in range(100)]
    a, b, c = split_words(words)
    assert (len(a), len(b), len(c)) == (80, 10, 10)
    assert split_words(words) == (a, b, c)
    assert sorted(a + b + c) == sorted(words)


def test_default_names_file_ships_inside_the_package() -> None:
    assert DEFAULT_NAMES.parent.name == "charlm" and DEFAULT_NAMES.is_file()


def test_names_file_loads() -> None:
    words = load_words()
    assert len(words) > 30000
    assert len(build_vocab(words)[0]) == 27
