from django.test import TestCase

from ..sparsamp import full_encode, full_decode


class SparSampTest(TestCase):

    def test_short_text(self):
        context = "Once upon a time"
        text = "attack@dawn"
        seed = 12345
        messages = full_encode(context, text, seed)
        decoded_text = full_decode(context, messages, seed)
        self.assertEqual(decoded_text, text)

    def test_middle_text(self):
        context = "Once upon a time"
        text = "Star Wars is an epic space opera franchise created by George Lucas. Its story takes place a long time ago in a galaxy far, far away and centers on the conflict between good and evil—a tale of Jedi Knights and Sith Lords, droids and aliens, and the rise and fall of empires."
        seed = 12345
        messages = full_encode(context, text, seed)
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
