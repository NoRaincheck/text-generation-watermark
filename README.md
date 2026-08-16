## Approach: Invisible Content Hash via Token-Level Watermarking

The core idea is to encode a secret hash into generated text by subtly biasing token choices, without making the output look unnatural. The hash is *invisible* (statistical, not visible markers) and *split-invariant* (detectable on any sub-span of the text).

### High-Level Algorithm

**1. Token Coloring (Deterministic Hashing)**

Each token ID is independently classified as "green" or "red" by hashing the token ID with a secret seed. The hash of `(seed, token_id)` produces a single bit — 1 means green, 0 means red. Crucially, this classification depends *only* on the token ID itself, not on its position in the text or the surrounding context.

**2. Seeded Generation (Encoding the Hash)**

During text generation, at each step the model produces a distribution over the vocabulary. Among the top candidates (filtered via nucleus sampling), those with near-equal logit scores are identified. For these ambiguous choices, green tokens receive a score boost, making them more likely to be sampled. Tokens that are clearly best (far above the competition) are chosen regardless of color.

This means the watermark only influences decisions the model was already uncertain about — the output remains natural-looking.

**3. Statistical Detection (Reading the Hash)**

To verify a passage contains the watermark, count the fraction of green tokens and compare it against the expected 50% baseline using a z-score test. A significantly elevated green fraction (z ≥ 1.65 for ~95% confidence) indicates the secret key was used during generation.

**4. Split Invariance**

Because token coloring depends only on the token ID (not position or context), any contiguous sub-span of text can be independently checked. The z-score detection works on sentence fragments, paragraphs, or any portion of the passage — the watermark signal persists regardless of where you start reading.

### Key Properties

- **Invisible**: No visible markers, delimiters, or artifacts — the text reads normally
- **Keyed**: Only someone with the secret seed can detect the watermark; wrong keys show no signal
- **Split-invariant**: Detection works on any sub-span, enabling paragraph-level verification
- **Reproducible**: Same seed + same prompt always produces the same output
- **Natural**: Watermark bias only applies to near-equal token choices, preserving fluency

### Sample Output

```sh
$ uv run watermark.py

=== watermarked output ===
Gravity is a fundamental force of nature that attracts two objects with mass toward each other. It is a universal force that shapes the structure of the universe, from the smallest subatomic particles to the vast expanses of galaxies. Gravity is responsible for many everyday phenomena, such as the falling of objects, the orbits of planets, and the stability of the Earth's shape.

In the context of the universe, gravity is what causes matter to clump together, forming stars, planets, and galaxies. It also plays a crucial role in the evolution of the cosmos, influencing the formation of cosmic structures and the distribution of matter.

There are several types of gravity, including:

* Newtonian gravity: A classical theory that describes the force of gravity as a universal constant, acting between all objects with mass.
* General relativity: A theory developed by Albert Einstein that describes gravity as the curvature of spacetime caused by massive objects.
* Dark matter: A type of matter that does not

=== negative-seed output ===
Gravity is a fundamental force of nature that attracts any two objects with mass toward each other. It is one of the four fundamental forces of nature, alongside electromagnetism, the strong nuclear force, and the weak nuclear force. Gravity is responsible for the motion of planets, the orbits of comets, and the collapse of stars. It also affects the trajectory of projectiles, the movement of objects in space, and the formation of structures like galaxies and solar systems. In general relativity, gravity is described as the curvature of spacetime caused by mass and energy, which influences the motion of objects within it.

=== plain output (baseline) ===
Gravity is a fundamental force of nature that attracts two objects with mass toward each other. It is a universal force that governs the motion of planets, stars, and other celestial bodies, as well as the behavior of objects on Earth. Gravity is responsible for the orbits of planets around the Sun, the tides on Earth, and the stability of the Earth's orbit around the Sun.

In physics, gravity is described by Newton's law of universal gravitation, which states that every mass attracts every other mass with a force proportional to the product of their masses and inversely proportional to the square of the distance between them. This law explains many phenomena, including the motion of comets, asteroids, and planets, as well as the behavior of objects in space.

Gravity also plays a crucial role in the formation of galaxies and the structure of the universe on large scales. It is a key factor in the way stars form and evolve, and it continues to shape the cos

=== detection ===
watermarked, correct key  : Hash(134/200 frac=0.670 z=+4.8)
watermarked, wrong key    : Hash(91/200 frac=0.455 z=-1.3)
negative-seed, correct key: Hash(71/126 frac=0.563 z=+1.4)
plain, correct key        : Hash(109/200 frac=0.545 z=+1.3)

=== hash across splits of watermarked text ===
from tok | key                              neg key                         
      19 | Hash(127/181 frac=0.702 z=+5.4) Hash(82/181 frac=0.453 z=-1.3)
      46 | Hash(107/154 frac=0.695 z=+4.8) Hash(69/154 frac=0.448 z=-1.3)
      76 | Hash(87/124 frac=0.702 z=+4.5) Hash(60/124 frac=0.484 z=-0.4)
     102 | Hash(67/98 frac=0.684 z=+3.6) Hash(52/98 frac=0.531 z=+0.6)
     128 | Hash(50/72 frac=0.694 z=+3.3) Hash(40/72 frac=0.556 z=+0.9)

=== hash across splits of negative-seed text ===
from tok | key                              neg key                         
      20 | Hash(45/106 frac=0.425 z=-1.6) Hash(64/106 frac=0.604 z=+2.1)
      48 | Hash(33/78 frac=0.423 z=-1.4) Hash(48/78 frac=0.615 z=+2.0)
      70 | Hash(24/56 frac=0.429 z=-1.1) Hash(35/56 frac=0.625 z=+1.9)
      97 | Hash(11/29 frac=0.379 z=-1.3) Hash(19/29 frac=0.655 z=+1.7)

=== hash across splits of plain text ===
from tok | key                              neg key                         
      19 | Hash(102/181 frac=0.564 z=+1.7) Hash(70/181 frac=0.387 z=-3.0)
      49 | Hash(85/151 frac=0.563 z=+1.5) Hash(59/151 frac=0.391 z=-2.7)
      80 | Hash(65/120 frac=0.542 z=+0.9) Hash(47/120 frac=0.392 z=-2.4)
     128 | Hash(42/72 frac=0.583 z=+1.4) Hash(27/72 frac=0.375 z=-2.1)
     156 | Hash(25/44 frac=0.568 z=+0.9) Hash(18/44 frac=0.409 z=-1.2)
     180 | Hash(12/20 frac=0.600 z=+0.9) Hash(10/20 frac=0.500 z=+0.0)
```

```sh
$ uv run watermark_synthid.py

=== watermarked output (5 layers, 2 competitors/match) ===
Gravity is a fundamental force of nature that attracts any two objects with mass toward each other. It is a universal constant, meaning it exists everywhere in the universe. Gravity is responsible for the motion of planets, the orbits of comets, and the stability of Earth's systems. It also plays a crucial role in the behavior of celestial bodies like black holes and neutron stars.

There are different types of gravity, including gravitational attraction, which pulls objects together, and gravitational repulsion, which pushes them apart. Gravity can also act on smaller objects, like water or even atoms themselves, though its effect on microscopic scales is not as significant.

Gravity is not only important on Earth but also in the vast emptiness of space. It shapes the structure of galaxies, the movement of stars within them, and even the way we perceive the universe. While we can't directly measure or manipulate gravity in everyday life, its effects are immeasurable in their impact on the cos

=== negative-seed output ===
Gravity is a fundamental force of nature that attracts two masses toward each other. It is a universally present force that governs the motion of planets, stars, and other large bodies in the universe. The strength of gravity depends on the masses involved and the distance between them.

According to Isaac Newton's Law of Universal Gravitation, the force of gravity between two masses is directly proportional to the product of their masses and inversely proportional to the square of the distance between them. This law explains many phenomena, such as the orbits of planets, the fall of an object to the ground, and the movement of satellites around Earth.

Gravity has been extensively studied in physics and has been confirmed through numerous experiments and observations. It is an essential component of the overall structure and stability of the universe, playing a critical role in the formation and evolution of galaxies, stars, and other cosmic structures.

=== plain output (baseline) ===
Gravity is a fundamental force of nature that attracts two objects with mass toward each other. It is a universal force that governs the motion of planets, stars, and other celestial bodies, as well as the behavior of objects on Earth. Gravity is responsible for the orbits of planets around the Sun, the tides on Earth, and the stability of the Earth's orbit around the Sun.

In physics, gravity is described by Newton's law of universal gravitation, which states that every mass attracts every other mass with a force proportional to the product of their masses and inversely proportional to the square of the distance between them. This law explains many phenomena, including the motion of comets, asteroids, and planets, as well as the behavior of objects in space.

Gravity also plays a crucial role in the formation of galaxies and the structure of the universe on large scales. It is a key factor in the way stars form and evolve, and it continues to shape the cos

=== detection ===
watermarked, correct key  : Hash(568/1000 frac=0.568 z=+4.3)
watermarked, wrong key    : Hash(489/1000 frac=0.489 z=-0.7)
negative-seed, correct key: Hash(468/910 frac=0.514 z=+0.9)
plain, correct key        : Hash(500/1000 frac=0.500 z=+0.0)

=== hash across splits of watermarked text ===
from tok | seed                             wrong seed                      
      20 | Hash(522/900 frac=0.580 z=+4.8) Hash(437/900 frac=0.486 z=-0.9)
      34 | Hash(476/830 frac=0.573 z=+4.2) Hash(401/830 frac=0.483 z=-1.0)
      58 | Hash(416/710 frac=0.586 z=+4.6) Hash(333/710 frac=0.469 z=-1.7)
      77 | Hash(359/615 frac=0.584 z=+4.2) Hash(283/615 frac=0.460 z=-2.0)
     104 | Hash(291/480 frac=0.606 z=+4.7) Hash(221/480 frac=0.460 z=-1.7)
     131 | Hash(201/345 frac=0.583 z=+3.1) Hash(170/345 frac=0.493 z=-0.3)
     151 | Hash(141/245 frac=0.576 z=+2.4) Hash(123/245 frac=0.502 z=+0.1)
     174 | Hash(81/130 frac=0.623 z=+2.8) Hash(60/130 frac=0.462 z=-0.9)

=== hash across splits of negative-seed text ===
from tok | seed                             wrong seed                      
      17 | Hash(426/825 frac=0.516 z=+0.9) Hash(494/825 frac=0.599 z=+5.7)
      41 | Hash(361/705 frac=0.512 z=+0.6) Hash(422/705 frac=0.599 z=+5.2)
      56 | Hash(318/630 frac=0.505 z=+0.2) Hash(378/630 frac=0.600 z=+5.0)
      97 | Hash(211/425 frac=0.496 z=-0.1) Hash(249/425 frac=0.586 z=+3.5)
     127 | Hash(138/275 frac=0.502 z=+0.1) Hash(162/275 frac=0.589 z=+3.0)
     147 | Hash(86/175 frac=0.491 z=-0.2) Hash(98/175 frac=0.560 z=+1.6)

=== hash across splits of plain text ===
from tok | seed                             wrong seed                      
      19 | Hash(457/905 frac=0.505 z=+0.3) Hash(478/905 frac=0.528 z=+1.7)
      49 | Hash(379/755 frac=0.502 z=+0.1) Hash(405/755 frac=0.536 z=+2.0)
      80 | Hash(302/600 frac=0.503 z=+0.2) Hash(316/600 frac=0.527 z=+1.3)
     128 | Hash(175/360 frac=0.486 z=-0.5) Hash(183/360 frac=0.508 z=+0.3)
     156 | Hash(106/220 frac=0.482 z=-0.5) Hash(118/220 frac=0.536 z=+1.1)
     180 | Hash(44/100 frac=0.440 z=-1.2) Hash(53/100 frac=0.530 z=+0.6)
```
