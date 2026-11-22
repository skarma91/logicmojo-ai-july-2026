# Data source and license

`guides.jsonl` holds 1,278 text chunks from the English Wikivoyage guides for 12 cities
(Lisbon, Kyoto, Mexico City, Cape Town, Istanbul, Hanoi, Edinburgh, Buenos Aires,
Marrakech, Vancouver, Prague, Singapore), sections Understand, Get around, See, Do, Eat,
Drink, and Stay safe. It was downloaded with `fetch_data.py` through the public MediaWiki
API.

Text: Wikivoyage contributors, https://en.wikivoyage.org, licensed CC BY-SA 4.0
(https://creativecommons.org/licenses/by-sa/4.0/). Each chunk keeps the URL of its page.
Anything you share that is built from this text must credit Wikivoyage and use the same
license. Guides change; facts such as opening hours and prices may be out of date.
