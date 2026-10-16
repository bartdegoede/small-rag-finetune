from small_rag_finetune.normalize import (
    go_blog,
    pep,
    rst_to_markdown,
    rust_blog,
    rust_rfc,
)


def write(tmp_path, rel, text):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def test_go_blog_reads_yaml_title_and_drops_template_actions(tmp_path):
    p = write(
        tmp_path,
        "go1.21.md",
        "---\ntitle: Go 1.21 is released!\ndate: 2023-08-08\nby:\n- Eli\n---\n\n"
        'Today we release Go 1.21.\n\n{{image "go1.21/gopher.png"}}\n\n'
        "## Tools\n\nFaster builds.\n",
    )
    doc = go_blog(p)
    assert doc.slug == "go1.21"
    assert doc.title == "Go 1.21 is released!"
    assert "{{" not in doc.body
    assert "## Tools" in doc.body and "Faster builds." in doc.body


def test_go_blog_without_front_matter_is_skipped(tmp_path):
    assert go_blog(write(tmp_path, "x.md", "no front matter")) is None


def test_rust_blog_reads_toml_and_prefixes_inside_rust(tmp_path):
    fm = (
        '+++\npath = "2025/12/05/x"\ntitle = "crates.io: Malicious crates"\n\n'
        '[extra]\nteam = "t"\n+++\n\n'
    )
    body = "## Summary\n\nTwo crates were removed.\n\n[socket]: https://socket.dev\n"
    top = rust_blog(write(tmp_path, "content/malicious.md", fm + body))
    inside = rust_blog(write(tmp_path, "content/inside-rust/team-update.md", fm + body))
    nested = rust_blog(write(tmp_path, "content/survey-2024/index.md", fm + body))
    assert top.slug == "malicious" and top.title == "crates.io: Malicious crates"
    assert inside.slug == "inside-rust-team-update"
    assert nested.slug == "survey-2024"
    assert "https://socket.dev" not in top.body
    assert "Two crates were removed." in top.body


def test_rust_blog_skips_section_index(tmp_path):
    p = write(tmp_path, "content/_index.md", '+++\ntitle = "x"\n+++\n')
    assert rust_blog(p) is None


def test_rust_rfc_titles_from_filename_and_drops_metadata(tmp_path):
    p = write(
        tmp_path,
        "text/2094-nll.md",
        "- Feature Name: nll\n- Start Date: 2017-08-02\n"
        "- RFC PR: [rust-lang/rfcs#2094](https://github.com/rust-lang/rfcs/pull/2094)\n\n"
        "## Summary\n[summary]: #summary\n\nExtend the borrow checker.\n\n"
        "- Note: a list item in the body stays.\n",
    )
    doc = rust_rfc(p)
    assert doc.slug == "2094-nll"
    assert doc.title == "RFC 2094: nll"
    assert "Start Date" not in doc.body and "#summary" not in doc.body
    assert "Extend the borrow checker." in doc.body
    assert "- Note: a list item in the body stays." in doc.body


def test_rust_rfc_skips_non_rfc_files(tmp_path):
    assert rust_rfc(write(tmp_path, "text/README.md", "# hi")) is None


PEP_TEXT = """PEP: 9999
Title: Example Things
Author: A <a@b.c>,
        B <b@c.d>
Status: Final

Abstract
========

This uses ``x := 1`` and :pep:`8` and `a link <https://x.org>`_.

Rationale
=========

Example::

    code here
    more code

Details
-------

Text after.

.. note::

   Dropped note.

Final line.
"""


def pep_body() -> str:
    return rst_to_markdown(PEP_TEXT.split("\n\n", 1)[1])


def test_pep_title_and_header_dropped(tmp_path):
    doc = pep(write(tmp_path, "peps/pep-9999.rst", PEP_TEXT))
    assert doc.slug == "pep-9999"
    assert doc.title == "PEP 9999: Example Things"
    assert "Author" not in doc.body and "B <b@c.d>" not in doc.body


def test_rst_headings_become_markdown_levels_in_order_of_appearance():
    lines = pep_body().splitlines()
    assert "# Abstract" in lines
    assert "# Rationale" in lines
    assert "## Details" in lines


def test_rst_inline_markup_and_blocks():
    md = pep_body()
    assert "This uses `x := 1` and PEP 8 and a link." in md
    assert "Example:" in md.splitlines()
    assert "code here" not in md
    assert "Dropped note." not in md
    assert "Text after." in md and "Final line." in md


def test_overlined_title():
    md = rst_to_markdown("=====\nTitle\n=====\n\nBody.\n")
    assert md.splitlines()[0] == "# Title"


def test_non_pep_file_is_skipped(tmp_path):
    assert pep(write(tmp_path, "peps/conf.rst", "x")) is None


def test_rust_rfc_metadata_after_a_title_line(tmp_path):
    p = write(
        tmp_path,
        "text/2947-crates-io-token-scopes.md",
        "## crates.io token scopes RFC\n\n- Feature Name: `scopes`\n"
        "- Start Date: 2020-06-24\n- crates.io issue: [x](https://x)\n\n"
        "# Summary\n\nTokens get scopes.\n\n- Note: kept.\n",
    )
    body = rust_rfc(p).body
    assert "## crates.io token scopes RFC" in body
    assert "Start Date" not in body and "crates.io issue" not in body
    assert "Tokens get scopes." in body and "- Note: kept." in body


def test_rst_simple_table_borders_and_footnotes_are_dropped():
    md = rst_to_markdown(
        "Text with a note [2]_ and a cite [#]_.\n\n"
        "======  ========\nNumber  Constant\n======  ========\n1       MODE_ECB\n"
        "======  ========\n"
    )
    assert "[2]_" not in md and "[#]_" not in md
    assert "Text with a note and a cite." in md
    assert "======" not in md
    assert "1       MODE_ECB" in md


def test_rst_roles_spanning_lines_and_with_targets():
    md = rst_to_markdown(
        "See the :pep:`style guidelines for the C code\nin CPython <7>` for more.\n"
    )
    assert md == "See the style guidelines for the C code\nin CPython for more."


def test_heading_anchor_attributes_are_stripped(tmp_path):
    go = write(
        tmp_path, "g.md", "---\ntitle: T\n---\n\n## Builds for Go {#go}\n\nText.\n"
    )
    rb = write(
        tmp_path, "content/r.md", '+++\ntitle = "T"\n+++\n\n## Why {#why}\n\nText.\n'
    )
    assert "## Builds for Go\n" in go_blog(go).body
    assert "## Why\n" in rust_blog(rb).body
