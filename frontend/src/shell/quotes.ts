/**
 * Thirty-one lines for the sidebar, one per day of the month.
 *
 * Every entry names its speaker and where the line is from. Nothing here is
 * "attributed": the earlier draft carried a disputed da Vinci and a
 * paraphrased Galileo, and a product whose pitch is that every number has a
 * source should not open its sidebar with a quotation that does not.
 *
 * The sidebar shows the speaker only (`DailyQuote.tsx`); `source` is the
 * provenance record behind that name, kept so the rule above stays checkable
 * (`quotes.test.ts`) and surfaced as the figure's tooltip, not as a caption.
 */
export interface Quote {
  text: string
  by: string
  /** Where the line comes from — a speech, a book, a mission. Not rendered as text. */
  source: string
}

export const QUOTES: readonly Quote[] = [
  {
    text: 'There are some who question the relevance of space activities in a developing nation. To us, there is no ambiguity of purpose.',
    by: 'Vikram Sarabhai',
    source: 'founding ISRO, 1968',
  },
  {
    text: 'We must be second to none in the application of advanced technologies to the real problems of man and society.',
    by: 'Vikram Sarabhai',
    source: 'founding ISRO, 1968',
  },
  {
    text: "Look again at that dot. That's here. That's home. That's us.",
    by: 'Carl Sagan',
    source: 'Pale Blue Dot, 1994',
  },
  {
    text: 'Imagination will often carry us to worlds that never were. But without it we go nowhere.',
    by: 'Carl Sagan',
    source: 'Cosmos, 1980',
  },
  {
    text: 'The cosmos is within us. We are made of star-stuff. We are a way for the universe to know itself.',
    by: 'Carl Sagan',
    source: 'Cosmos, 1980',
  },
  {
    text: 'Saare jahan se achha.',
    by: 'Rakesh Sharma',
    source: 'from orbit, asked how India looked, 1984',
  },
  {
    text: 'Look at the sky. We are not alone. The whole universe is friendly to us and conspires only to give the best to those who dream and work.',
    by: 'A. P. J. Abdul Kalam',
    source: 'Wings of Fire, 1999',
  },
  {
    text: 'The path from dreams to success does exist. May you have the vision to find it, the courage to get on to it, and the perseverance to follow it.',
    by: 'Kalpana Chawla',
    source: 'letter to students, 2000',
  },
  {
    text: 'Earth is the cradle of humanity, but one cannot live in a cradle forever.',
    by: 'Konstantin Tsiolkovsky',
    source: 'letter, 1911',
  },
  {
    text: 'We came all this way to explore the Moon, and the most important thing is that we discovered the Earth.',
    by: 'William Anders',
    source: 'Apollo 8, 1968',
  },
  {
    text: "The stars don't look bigger, but they do look brighter.",
    by: 'Sally Ride',
    source: 'on orbit, STS-7, 1983',
  },
  {
    text: "Never be limited by other people's limited imaginations.",
    by: 'Mae Jemison',
    source: 'commencement address',
  },
  {
    text: 'Magnificent desolation.',
    by: 'Buzz Aldrin',
    source: 'Apollo 11, on the lunar surface, 1969',
  },
  {
    text: 'When you look at the stars and the galaxy, you feel that you are not just from any particular piece of land, but from the solar system.',
    by: 'Kalpana Chawla',
    source: 'interview, 1997',
  },
  {
    text: 'You develop an instant global consciousness, a people orientation, an intense dissatisfaction with the state of the world, and a compulsion to do something about it.',
    by: 'Edgar Mitchell',
    source: 'Apollo 14, interview, 1974',
  },
  {
    text: 'Orbiting Earth in the spaceship, I saw how beautiful our planet is. People, let us preserve and increase this beauty, not destroy it.',
    by: 'Yuri Gagarin',
    source: 'Vostok 1, 1961',
  },
  {
    text: "That's one small step for a man, one giant leap for mankind.",
    by: 'Neil Armstrong',
    source: 'Apollo 11, 1969',
  },
  {
    text: 'Equipped with his five senses, man explores the universe around him and calls the adventure Science.',
    by: 'Edwin Hubble',
    source: 'The Nature of Science, 1954',
  },
  {
    text: 'Any sufficiently advanced technology is indistinguishable from magic.',
    by: 'Arthur C. Clarke',
    source: 'Profiles of the Future, 1973 edition',
  },
  {
    text: 'Look up at the stars and not down at your feet. Try to make sense of what you see, and wonder about what makes the universe exist.',
    by: 'Stephen Hawking',
    source: 'interview, 2010',
  },
  {
    text: 'The universe is under no obligation to make sense to you.',
    by: 'Neil deGrasse Tyson',
    source: 'Astrophysics for People in a Hurry, 2017',
  },
  {
    text: 'If I have seen further it is by standing on the shoulders of giants.',
    by: 'Isaac Newton',
    source: 'letter to Robert Hooke, 1675',
  },
  {
    text: 'The universe cannot be read until we have learned the language and become familiar with the characters in which it is written. It is written in mathematical language.',
    by: 'Galileo Galilei',
    source: 'Il Saggiatore, 1623',
  },
  {
    text: 'The Earth is a very small stage in a vast cosmic arena.',
    by: 'Carl Sagan',
    source: 'Pale Blue Dot, 1994',
  },
  {
    text: 'It is science alone that can solve the problems of hunger and poverty, of insanitation and illiteracy, of superstition and deadening custom and tradition.',
    by: 'Jawaharlal Nehru',
    source: 'Indian Science Congress, 1937',
  },
  {
    text: "Space is for everybody. It's not just for a few people in science or math, or for a select group of astronauts.",
    by: 'Christa McAuliffe',
    source: 'interview, 1985',
  },
  {
    text: 'The first principle is that you must not fool yourself — and you are the easiest person to fool.',
    by: 'Richard Feynman',
    source: 'Caltech commencement, 1974',
  },
  {
    text: 'The Earth was small, light blue, and so touchingly alone, our home that must be defended like a holy relic.',
    by: 'Aleksei Leonov',
    source: 'Voskhod 2, the first spacewalk, 1965',
  },
  {
    text: 'I really believe that if the political leaders of the world could see their planet from a distance of 100,000 miles their outlook could be fundamentally changed.',
    by: 'Michael Collins',
    source: 'Carrying the Fire, 1974',
  },
  {
    text: 'Across the sea of space, the stars are other suns.',
    by: 'Carl Sagan',
    source: 'Cosmos, 1980',
  },
  {
    text: 'Astronomy compels the soul to look upward, and leads us from this world to another.',
    by: 'Plato',
    source: 'The Republic, Book VII',
  },
]
