from enum import Enum


def create(genre: str):
    if genre == "love_stories":
        return Genre.LOVE_STORIES
    elif genre == "ghost_stories":
        return Genre.GHOST_STORIES
    elif genre == "science_fiction":
        return Genre.SCIENCE_FICTION
    else:
        return Genre.LOVE_STORIES


class Genre(Enum):
    LOVE_STORIES = "love_stories"
    GHOST_STORIES = "ghost_stories"
    SCIENCE_FICTION = "science_fiction"

    def __str__(self):
        name = self.name.replace("_", " ").title()
        return name.replace(" Stories", " stories")

