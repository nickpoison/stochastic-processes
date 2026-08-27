"""
Minimal pure-Python reader for R's .rda/.RData serialization format
(the "RDX2"/"RDX3" XDR binary format written by R's save()).

This implements just enough of R's serialization spec to read the
simple objects found in the astsa data package: numeric/integer/character
vectors, "ts" objects (numeric vector + tsp/class attrs), data.frames,
and factors. It is NOT a general RDS/RDA reader.

Reference: R Internals manual, "Serialization Formats" section, and
R source src/main/serialize.c (WriteItem/ReadItem, SEXPTYPE constants).
"""
import struct
import gzip

# --- SEXPTYPE constants (subset we need) ---
NILSXP = 0
SYMSXP = 1
LISTSXP = 2
CLOSXP = 3
ENVSXP = 4
PROMSXP = 5
LANGSXP = 6
CHARSXP = 9
LGLSXP = 10
INTSXP = 13
REALSXP = 14
CPLXSXP = 15
STRSXP = 16
VECSXP = 19
EXPRSXP = 20
S4SXP = 25

REFSXP = 255
NILVALUE_SXP = 254
GLOBALENV_SXP = 253
UNBOUNDVALUE_SXP = 252
MISSINGARG_SXP = 251
BASENAMESPACE_SXP = 250
NAMESPACESXP = 249
PACKAGESXP = 248
PERSISTSXP = 247
CLASSREFSXP = 246
GENERICREFSXP = 245
BASEENV_SXP = 241
EMPTYENV_SXP = 242

NA_INTEGER = -2147483648


class RSymbol(str):
    """A distinguishable subclass so we can tell symbols from strings."""
    pass


class RObject:
    """Generic holder for a parsed R value with optional attributes."""
    def __init__(self, value, attrs=None, rtype=None):
        self.value = value
        self.attrs = attrs or {}
        self.rtype = rtype

    def __repr__(self):
        return f"RObject(rtype={self.rtype}, len={len(self.value) if hasattr(self.value,'__len__') else '?'}, attrs={list(self.attrs.keys())})"


class RReader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.ref_table = []

    # -- low level --
    def _bytes(self, n):
        b = self.data[self.pos:self.pos + n]
        if len(b) != n:
            raise EOFError("unexpected end of data")
        self.pos += n
        return b

    def _int(self):
        return struct.unpack('>i', self._bytes(4))[0]

    def _double(self):
        return struct.unpack('>d', self._bytes(8))[0]

    def read_header(self):
        line1 = self._read_line()  # "RDX2" or "RDX3"
        line2 = self._read_line()  # "X" (xdr) -- we only support X
        assert line2 == "X", f"unsupported encoding {line2!r}"
        version = self._int()
        writer_version = self._int()
        min_reader_version = self._int()
        if version == 3:
            # version 3 adds a native encoding name string right after the header ints
            nchar = self._int()
            self._bytes(nchar)  # e.g. b"UTF-8"
        return version

    def _read_line(self):
        start = self.pos
        idx = self.data.index(b"\n", start)
        s = self.data[start:idx].decode("ascii")
        self.pos = idx + 1
        return s

    # -- item dispatch --
    def read_item(self):
        flags = self._int()
        return self._read_with_flags(flags)

    def _read_with_flags(self, flags):
        type_ = flags & 255

        if type_ == NILVALUE_SXP or type_ == NILSXP:
            return None
        if type_ == REFSXP:
            idx = flags >> 8
            if idx == 0:
                idx = self._int()
            return self.ref_table[idx - 1]
        if type_ in (GLOBALENV_SXP, BASEENV_SXP, EMPTYENV_SXP, MISSINGARG_SXP,
                     UNBOUNDVALUE_SXP, BASENAMESPACE_SXP):
            return None

        isobj = bool(flags & (1 << 8))
        hasattr_ = bool(flags & (1 << 9))
        hastag = bool(flags & (1 << 10))

        if type_ == SYMSXP:
            name = self.read_item()  # CHARSXP -> python str
            sym = RSymbol(name)
            self.ref_table.append(sym)
            return sym

        if type_ == CHARSXP:
            length = self._int()
            if length == -1:
                return None
            raw = self._bytes(length)
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError:
                return raw.decode("latin1")

        if type_ in (LISTSXP, LANGSXP):
            attr = self.read_item() if hasattr_ else None
            tag = self.read_item() if hastag else None
            car = self.read_item()
            cdr = self.read_item()  # recurses into next pairlist node or None
            return {"tag": tag, "car": car, "cdr": cdr, "attr": attr}

        if type_ == INTSXP:
            n = self._int()
            vals = []
            for _ in range(n):
                v = self._int()
                vals.append(None if v == NA_INTEGER else v)
            attrs = self.read_item() if hasattr_ else None
            return RObject(vals, self._pairlist_to_dict(attrs), "int")

        if type_ == REALSXP:
            n = self._int()
            vals = [self._double() for _ in range(n)]
            attrs = self.read_item() if hasattr_ else None
            return RObject(vals, self._pairlist_to_dict(attrs), "real")

        if type_ == LGLSXP:
            n = self._int()
            vals = []
            for _ in range(n):
                v = self._int()
                vals.append(None if v == NA_INTEGER else bool(v))
            attrs = self.read_item() if hasattr_ else None
            return RObject(vals, self._pairlist_to_dict(attrs), "logical")

        if type_ == STRSXP:
            n = self._int()
            vals = [self.read_item() for _ in range(n)]
            attrs = self.read_item() if hasattr_ else None
            return RObject(vals, self._pairlist_to_dict(attrs), "str")

        if type_ == VECSXP:
            n = self._int()
            vals = [self.read_item() for _ in range(n)]
            attrs = self.read_item() if hasattr_ else None
            return RObject(vals, self._pairlist_to_dict(attrs), "list")

        raise NotImplementedError(f"Unsupported SEXP type {type_} (flags={flags})")

    @staticmethod
    def _pairlist_to_dict(node):
        d = {}
        while node is not None:
            name = str(node["tag"]) if node["tag"] is not None else None
            d[name] = node["car"]
            node = node["cdr"]
        return d

    def read_toplevel(self):
        """Top-level of a save() file is a tagged pairlist: name -> value, ..."""
        out = {}
        item = self.read_item()
        # item is either a dict-node (pairlist chain) or None (empty save)
        node = item
        while isinstance(node, dict):
            name = str(node["tag"])
            out[name] = node["car"]
            node = node["cdr"]
        return out


def load_rda(path):
    """Load a .rda file, return dict of {varname: RObject-or-python-value}."""
    with open(path, "rb") as f:
        raw = f.read()
    try:
        raw = gzip.decompress(raw)
    except OSError:
        pass  # not gzip-compressed
    reader = RReader(raw)
    reader.read_header()
    return reader.read_toplevel()
