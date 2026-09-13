"""Generate movies.jsonl: a small, real movie-plot corpus for the micro-assignments.

This is the micro-assignment dataset, deliberately DIFFERENT from the IRS
tax-publication corpus used in the class notebooks, so the micro is a genuine
transfer of the taught pattern to new data. Facts (year, director, genre) are
accurate; the plot blurbs are short factual summaries in our own words, with the
Wikipedia page as the source. `fetch_movies.py` builds the full set from Wikipedia.

Run:  python make_movies.py   ->  writes movies.jsonl next to this file

Record schema (one JSON object per line):
  id, title, year, genre, director, source_url, plot
"""

import json
import pathlib

# (id, title, year, genre, director, plot). Note the two "The Italian Job"
# entries (1969 and 2003): a remake pair, so a year filter must pick the version
# you want, just like the tax-year filter in the class corpus.
RECORDS = [
    ("inception-2010", "Inception", 2010, "Science Fiction", "Christopher Nolan",
     "A skilled thief who steals secrets by entering people's dreams is offered a chance to erase his record by planting an idea in a target's mind, a task called inception."),
    ("matrix-1999", "The Matrix", 1999, "Science Fiction", "The Wachowskis",
     "A hacker learns that reality is a simulation created by machines and joins a rebellion to free humanity from the illusion."),
    ("italian-job-1969", "The Italian Job", 1969, "Crime", "Peter Collinson",
     "A crew executes an audacious gold heist in Turin, using a traffic jam and three Mini Cooper cars to escape."),
    ("italian-job-2003", "The Italian Job", 2003, "Crime", "F. Gary Gray",
     "After a betrayal, a team of thieves plans an elaborate gold heist in Los Angeles, again using Mini Cooper cars to slip through the city."),
    ("oceans-eleven-2001", "Ocean's Eleven", 2001, "Crime", "Steven Soderbergh",
     "A recently paroled con man assembles a team of eleven specialists to rob three Las Vegas casinos on the same night."),
    ("shawshank-1994", "The Shawshank Redemption", 1994, "Drama", "Frank Darabont",
     "A banker wrongly convicted of murder forms a friendship in prison and quietly works toward an escape over many years."),
    ("toy-story-1995", "Toy Story", 1995, "Animation", "John Lasseter",
     "A cowboy doll feels threatened when a flashy new spaceman toy arrives, and the rivals must work together to find their way home."),
    ("finding-nemo-2003", "Finding Nemo", 2003, "Animation", "Andrew Stanton",
     "A timid clownfish crosses the ocean to rescue his son, who has been captured and placed in a dentist's aquarium."),
    ("jurassic-park-1993", "Jurassic Park", 1993, "Science Fiction", "Steven Spielberg",
     "Scientists clone dinosaurs for a theme park, but the animals break loose when the park's security fails."),
    ("dark-knight-2008", "The Dark Knight", 2008, "Action", "Christopher Nolan",
     "A masked vigilante confronts an anarchic criminal who plunges the city into chaos and forces impossible moral choices."),
    ("titanic-1997", "Titanic", 1997, "Romance", "James Cameron",
     "A poor artist and a wealthy young woman fall in love aboard the doomed ocean liner on its first and final voyage."),
    ("blade-runner-1982", "Blade Runner", 1982, "Science Fiction", "Ridley Scott",
     "A weary detective hunts fugitive bioengineered humans called replicants in a rain-soaked future Los Angeles."),
    ("interstellar-2014", "Interstellar", 2014, "Science Fiction", "Christopher Nolan",
     "As Earth becomes unlivable, a former pilot leads a mission through a wormhole to find a new home for humanity."),
    ("godfather-1972", "The Godfather", 1972, "Crime", "Francis Ford Coppola",
     "The reluctant youngest son of a crime family is drawn into the business and gradually becomes its ruthless leader."),
    ("spirited-away-2001", "Spirited Away", 2001, "Animation", "Hayao Miyazaki",
     "A young girl wanders into a spirit world and must work in a bathhouse to free her parents and find her way back."),
    ("parasite-2019", "Parasite", 2019, "Drama", "Bong Joon-ho",
     "A poor family schemes to be employed by a wealthy household, until a hidden secret upends the arrangement."),
    ("mad-max-fury-road-2015", "Mad Max: Fury Road", 2015, "Action", "George Miller",
     "In a desert wasteland, a drifter and a rebel commander flee a tyrant across the sands in a relentless vehicle chase."),
    ("coco-2017", "Coco", 2017, "Animation", "Lee Unkrich",
     "A boy who dreams of music is transported to the Land of the Dead, where he seeks a famous ancestor and his family's blessing."),
]


def main():
    out = pathlib.Path(__file__).with_name("movies.jsonl")
    with out.open("w") as f:
        for mid, title, year, genre, director, plot in RECORDS:
            f.write(json.dumps({
                "id": mid, "title": title, "year": year, "genre": genre,
                "director": director,
                "source_url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                "plot": plot,
            }) + "\n")
    print(f"wrote {len(RECORDS)} movies to {out}")


if __name__ == "__main__":
    main()
