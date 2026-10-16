from small_rag_finetune.normalize import go_blog, rust_blog, rust_rfc


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
