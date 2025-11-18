from django.test import TestCase

from ..encoding import full_encode
from ..decoding import full_decode
from ..token_graph import build_token_graph
from ..sparsamp_utils import TOKENIZER

class SparSampTest(TestCase):

    def test_short_text(self):
        context = "Once upon a time"
        text = "attack@dawn"
        seed = 12345
        messages = full_encode(context, text, seed)
        build_token_graph(text, TOKENIZER)
        decoded_text = full_decode(context, messages, seed)
        self.assertEqual(decoded_text, text)

    def test_middle_text(self):
        context = "Once upon a time"
        text = "Star Wars is an epic space opera franchise created by George Lucas. Its story takes place a long time ago in a galaxy far, far away and centers on the conflict between good and evil—a tale of Jedi Knights and Sith Lords, droids and aliens, and the rise and fall of empires."
        seed = 12345
        messages = full_encode(context, text, seed)
        decoded_text = full_decode(context, messages, seed)
        self.assertEqual(text, decoded_text)

    def test_firefox_text(self):
        context= "Steganography is the practice of hiding messages within other non-secret text or data."
        text = "Firefox is a free, open-source web browser developed by Mozilla Foundation. Released in 2004, it challenged Internet Explorer's monopoly and continues to promote web standards and user privacy. Firefox offers robust security features including Enhanced Tracking Protection, which blocks third-party cookies and trackers by default. The browser supports extensive customization through add-ons and themes, allowing users to personalize their browsing experience. Available across Windows, macOS, Linux, Android, and iOS platforms, Firefox synchronizes bookmarks, passwords, and history between devices. Despite Chrome's market dominance, Firefox remains a vital alternative for users prioritizing privacy, open-source values, and control over their online data protection."
        seed = 12345
        messages = full_encode(context, text, seed)
        decoded_text = full_decode(context, messages, seed)
        self.assertEqual(decoded_text, text)

    def test_decoding_with_ta(self):
        context = "Once upon a time"
        text = "Star Wars is an epic space opera franchise created by George Lucas. Its story takes place a long time ago in a galaxy far, far away and centers on the conflict between good and evil—a tale of Jedi Knights and Sith Lords, droids and aliens, and the rise and fall of empires."
        seed = 12345
        messages = [" you seem to have trouble wrapping your mind around a problem. Each headache causes you to work out actionable solutions. Run, charter, read notes of 'Yhmump, run, aren't you getting frustrated?' Get out of your hurry and leapfrog until you have the energy, the motivation, the willpower, the mindful choice to fly. Soon you put a stop to any and all fear and build an inner source of discipline. Allow your body to resist wherever unlimited stress might endanger you.\n\nAnd", ", he could scarcely have minded the lack of consolation in neglecting to pay the cost of another year's rental — distances that rendered him an indispensable tinker or the porter of trips in these trials, thrown up with serious grief and ruin. But this present tail was full of satisfaction for sometimes a fortnight — in which case his partiality would have absolutely destroyed his sagacity; but even then not to scratch, an infinite while more. A great shyness struck and fed upon him on account of the hospitality of the summer", ', after nearly forty thousand years, King Alexander the Great created a city, Nevia, in the land of the sword. From this time, shortly before his Death, Alexander took refuge in Marganis in the middle of a Great War which occupied one of the years 1,000 BCE to 433 CE. With admittance, Alexander built his first roads. He then hated its geography, and gave it to Persia and changed it to Flowers, telling it that only true warriors would fight like that. (', ', the Nevada landscape presented a threat to the much coveted modern ranch each year. We see abundant biological diversity of different kinds roaming the landscape, from and to nest of deer in zippo lagoons to canines in douglas fir stumps—large, solitary mammals that can move through the gaps with no visible stripe of sunlight. Not to mention a recently discovered five-foot coal pipe found at one of the largest hog habitat areas in Nevada Parks and Country. Stripes bridging through the swamps proved', ' I would have resized too much but then I started looking closer and more and until I could see their busts, delicate breasts and posed bodies.\n\nIt is that view', ', Mary W. Snow recognized that every single strategy']
        decoded_text = full_decode(context, messages, seed)
        self.assertEqual(decoded_text, text)

    def test_long_text(self):
        context = "Once upon a time"
        text = """Star Wars is an epic space opera franchise created by George Lucas. Its story takes place a long time ago in a galaxy far, far away and centers on the conflict between good and evil—a tale of Jedi Knights and Sith Lords, droids and aliens, and the rise and fall of empires.

The saga began in 1977 with the release of Star Wars (later subtitled Episode IV: A New Hope). It quickly became a global phenomenon and laid the foundation for one of the most influential franchises in film history. The original trilogy tells the story of a farm boy named Luke Skywalker, the wise Obi-Wan Kenobi, Princess Leia, and the charming smuggler Han Solo, as they fight against the Galactic Empire ruled by the evil Emperor Palpatine and his fearsome enforcer, Darth Vader.

Beyond the original trilogy, the Star Wars story is expanded through a prequel trilogy that dives into the rise of Anakin Skywalker and his transformation into Darth Vader—focusing on the downfall of the Jedi Order and the rise of the Empire. The sequel trilogy, developed decades later, continues the story with a new generation of heroes and villains, building on classic themes of hope, redemption, destiny, and the ongoing battle between light and dark.



The universe is marked by advanced technology, such as starships that travel at lightspeed, planets of every imaginable environment, and incredible weapons like lightsabers and the planet-destroying Death Star. At the heart of the saga is the Force—an energy field created by all life that can be harnessed for good or evil, granting its users extraordinary abilities. This duality of the Force is embodied by its two great orders: the noble Jedi, who serve as peacekeepers, and the power-hungry Sith, who seek dominance over the galaxy.

The series spans centuries of galactic history—featuring epic battles, dramatic betrayals, powerful friendships, and moral dilemmas. Characters like Yoda, Chewbacca, R2-D2, C-3PO, Lando Calrissian, and many more have become cultural icons.

The impact of Star Wars extends well beyond the films. It includes animated and live-action TV series, novels, comics, video games, toys, and even theme parks, creating a vast expanded universe and a passionate fan community. Star Wars continues to inspire new generations with stories of courage, hope, and the belief that even the smallest person can change the fate of the galaxy.
"""
        seed = 12345
        messages = full_encode(context, text, seed)
        decoded_text = full_decode(context, messages, seed)
        self.assertEqual(decoded_text, text)
