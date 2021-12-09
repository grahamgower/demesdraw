import sys

import packaging.version


def is_new_stable(old_stable, tag):
    """
    Return 0 if `tag` should be the new 'stable' docs folder, 1 otherwise.
    """
    try:
        v_t = packaging.version.Version(tag)
    except packaging.version.InvalidVersion:
        return 1
    v_s = packaging.version.Version(old_stable)
    return 0 if not v_t.is_prerelease and v_t > v_s else 1


if __name__ == "__main__":
    assert is_new_stable("0.2.0", "taggy-mac-tag-tag") != 0
    assert is_new_stable("0.2.0", "0.1.0") != 0
    assert is_new_stable("0.2.0", "0.1.0b1") != 0
    assert is_new_stable("0.2.0", "0.2.0b1") != 0
    assert is_new_stable("0.2.0", "0.3.0b1") != 0
    assert is_new_stable("0.2.0", "0.2.1") == 0
    assert is_new_stable("0.2.0", "0.3.0") == 0

    if len(sys.argv) != 3:
        print(f"{sys.argv[0]} old_stable tag")
        exit(2)

    exit(is_new_stable(sys.argv[1], sys.argv[2]))
