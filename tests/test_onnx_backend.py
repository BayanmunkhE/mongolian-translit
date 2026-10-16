from translit.onnx_backend import decode, encode


def test_encode_is_byte_plus_three_with_eos():
    assert encode("ab") == [ord("a") + 3, ord("b") + 3, 1]
    assert encode("ө")[:-1] == [b + 3 for b in "ө".encode("utf-8")]


def test_decode_joins_labels_on_first_byte_and_copies_unknown():
    vocab = ["", "<unk>", "б", "айна", "ө"]
    # "bn" -> б + айна ; the second byte of a 2-byte char carries the empty label
    assert decode("bn", [2, 3], vocab) == "байна"
    assert decode("ө!", [4, 0, 1], vocab) == "ө!"  # continuation byte empty, "!" unknown -> copied
    assert decode("abc", [2], vocab) == "бbc"  # missing predictions count as unknown
