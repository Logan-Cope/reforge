"""Unit tests for the pure reference-image selection in llm/claude.py.

The identity-reference upgrade passes Logan's reference photos alongside each clip's
frames. `image_files` selects which files in the reference dir are usable images.
No anthropic SDK is imported at module load, so this stays a light pure test.
"""
from reforge.llm.claude import image_files, is_image_file


def test_is_image_file_accepts_common_image_extensions():
    assert is_image_file("gym_face_1.jpg")
    assert is_image_file("IMG_5923.JPG")     # case-insensitive
    assert is_image_file("a.jpeg")
    assert is_image_file("b.png")
    assert is_image_file("c.webp")


def test_is_image_file_rejects_non_images():
    assert not is_image_file("notes.txt")
    assert not is_image_file("clip.mov")
    assert not is_image_file("README")


def test_image_files_filters_and_sorts_deterministically():
    listing = ["identity_reference/b.png", "identity_reference/a.JPG",
               "identity_reference/notes.txt", "identity_reference/c.jpeg"]
    assert image_files(listing) == [
        "identity_reference/a.JPG",
        "identity_reference/b.png",
        "identity_reference/c.jpeg",
    ]
