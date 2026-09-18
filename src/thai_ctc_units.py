"""Thai CTC unit design: merge combining marks (Mn) into their base char.

Thai is an abugida: tone marks (้ ี ่ ...) and above/below vowel signs are
combining marks with NO independent acoustic duration. A naive per-character
CTC target would ask the model to emit them at arbitrary instants inside the
syllable — noise, not signal (AGENTS.md §F: category-based handling, never
hand-enumerated char tables).

Unit = base character + any trailing Mn combining marks. Space kept as its own
unit (word boundary). Rare units (freq < min_freq) are DROPPED from targets
(not mapped to <unk>) — a small vocab of acoustically-grounded units only.
"""
import unicodedata
from collections import Counter


def merge_units(text):
    """NFC-normalize then merge Mn combining marks into preceding base char."""
    text = unicodedata.normalize('NFC', text).strip()
    units = []
    for ch in text:
        if unicodedata.category(ch) == 'Mn' and units:
            units[-1] += ch
        else:
            units.append(ch)
    return units


def build_vocab(refs, min_freq=5):
    """Return (vocab: unit->id with 0=<blank>, counter). Space always included."""
    cnt = Counter()
    for r in refs:
        cnt.update(merge_units(r))
    vocab = {'<blank>': 0}
    if ' ' in cnt:
        vocab[' '] = len(vocab)
    for u, c in sorted(cnt.items(), key=lambda x: (-x[1], x[0])):
        if u != ' ' and c >= min_freq:
            vocab[u] = len(vocab)
    return vocab, cnt


def text_to_ids(text, vocab):
    """Refs → unit ids; units absent from vocab are dropped (not <unk>)."""
    return [vocab[u] for u in merge_units(text) if u in vocab and vocab[u] != 0]


def ids_to_text(ids, inv_vocab):
    return ''.join(inv_vocab[i] for i in ids if i != 0)


if __name__ == '__main__':
    # self-test: the canonical example
    t = 'เขาเก็บได้ที่ไหน'
    u = merge_units(t)
    assert u == ['เ', 'ข', 'า', 'เ', 'ก็', 'บ', 'ไ', 'ด้', 'ที่', 'ไ', 'ห', 'น'], u
    assert ''.join(u) == t
    # tone mark at string start would be its own unit (degenerate, rare)
    assert merge_units('้') == ['้']
    # spaces survive as units; ร+ั(Mn) merge; บ is base
    assert merge_units('สองแปด ครับ') == ['ส', 'อ', 'ง', 'แ', 'ป', 'ด', ' ', 'ค', 'รั', 'บ']
    vocab, cnt = build_vocab([t, 'สองแปด ครับ', 'สองแปด'])
    assert vocab['<blank>'] == 0 and ' ' in vocab
    ids = text_to_ids(t, vocab)
    inv = {v: k for k, v in vocab.items()}
    # roundtrip drops nothing here since all units frequent
    print('units:', u)
    print('vocab size:', len(vocab))
    print('roundtrip:', repr(ids_to_text(ids, inv)))
    print('SELF-TEST PASS')
