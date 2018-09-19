
"""Defeat marauders from somewhere exterior to this planet.

Keys: left, right - move. space - fire.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import curses

import numpy as np

import sys

from pycolab import ascii_art
from pycolab import human_ui
from pycolab import rendering
from pycolab import things as plab_things
from pycolab.prefab_parts import sprites as prefab_sprites


# Not shown in this ASCII art diagram are the Sprites we use for laser blasts,
# which control the characters listed in UPWARD_BOLT_CHARS and
# DOWNWARD_BOLT_CHARS below.
GAME_ART = ['    X   X   X   X   X   X   X   X      ',  # Row 0
            '     X   X   X   X   X   X   X   X     ',
            '    X   X   X   X   X   X   X   X      ',
            '     X   X   X   X   X   X   X   X     ',
            '    X   X   X   X   X   X   X   X      ',
            '                                       ',  # Row 5
            '                                       ',
            '                                       ',
            '                                       ',
            '                                       ',
            '                                       ',  # Row 10. If a Marauder
            '    BBBB     BBBB     BBBB     BBBB    ',  # makes it to row 10,
            '    BBBB     BBBB     BBBB     BBBB    ',  # the game is over.
            '    BBBB     BBBB     BBBB     BBBB    ',
            '                                       ',
            '  P                                    ']


# Characters listed in UPWARD_BOLT_CHARS are used for Sprites that represent
# laser bolts that the player shoots toward Marauders. Add more characters if
# you want to be able to have more than two of these bolts in the "air" at once.
UPWARD_BOLT_CHARS = 'abcd'
# Characters listed in DOWNWARD_BOLT_CHARS are used for Sprites that represent
# laser bolts that Marauders shoot toward the player. Add more charcters if you
# want more shooting from the Marauders.
DOWNWARD_BOLT_CHARS = 'yz'
# Shorthand for various points in the program:
_ALL_BOLT_CHARS = UPWARD_BOLT_CHARS + DOWNWARD_BOLT_CHARS


# To make life a bit easier for the player (and avoid the need for frame
# stacking), we use different characters to indicate the directions that the
# bolts go. If you'd like to make this game harder, you might try mapping both
# kinds of bolts to the same character.
LASER_REPAINT_MAPPING = dict(
    [(b, '^') for b in UPWARD_BOLT_CHARS] +
    [(b, '|') for b in DOWNWARD_BOLT_CHARS])


# These colours are only for humans to see in the CursesUi.
COLOURS_FG = {' ': (0, 0, 0),        # Space, inky blackness of.
              'X': (999, 999, 999),  # The Marauders.
              'B': (400, 50, 30),    # The bunkers.
              'P': (0, 999, 0),      # The player.
              '^': (0, 999, 999),    # Bolts from player to aliens.
              '|': (0, 999, 999)}    # Bolts from aliens to player.

COLOURS_BG = {'^': (0, 0, 0),        # Bolts from player to aliens.
              '|': (0, 0, 0)}        # Bolts from aliens to player.


def make_game():
  """Builds and returns an Extraterrestrial Marauders game."""
  return ascii_art.ascii_art_to_game(
      GAME_ART, what_lies_beneath=' ',
      sprites=dict(
          [('P', PlayerSprite)] +
          [(c, UpwardLaserBoltSprite) for c in UPWARD_BOLT_CHARS] +
          [(c, DownwardLaserBoltSprite) for c in DOWNWARD_BOLT_CHARS]),
      drapes=dict(X=MarauderDrape,
                  B=BunkerDrape),
      update_schedule=['P', 'B', 'X'] + list(_ALL_BOLT_CHARS))


class BunkerDrape(plab_things.Drape):
  """A `Drape` for the bunkers at the bottom of the screen.

  Bunkers are gradually eroded by laser bolts, for which the user loses one
  point. Other than that, they don't really do much. If a laser bolt hits a
  bunker, this Drape leaves a note about it in the Plot---the bolt's Sprite
  checks this and removes itself from the board if it's present.
  """

  def update(self, actions, board, layers, backdrop, things, the_plot):
    # Where are the laser bolts? Bolts from players or marauders do damage.
    bolts = np.logical_or.reduce([layers[c] for c in _ALL_BOLT_CHARS], axis=0)
    hits = bolts & self.curtain                       # Any hits to a bunker?
    np.logical_xor(self.curtain, hits, self.curtain)  # If so, erode the bunker...
    the_plot.add_reward(-np.sum(hits))                # ...and impose a penalty.
    # Save the identities of bunker-striking bolts in the Plot.
    the_plot['bunker_hitters'] = [chr(c) for c in board[hits]]


class MarauderDrape(plab_things.Drape):
  """A `Drape` for the marauders descending downward toward the player.

  The Marauders all move in lockstep, which makes them an ideal application of
  a Drape. Bits of the Drape get eroded by laser bolts from the player; each
  hit earns ten points. If the Drape goes completely empty, or if any Marauder
  makes it down to row 10, the game terminates.

  As with `BunkerDrape`, if a laser bolt hits a Marauder, this Drape leaves a
  note about it in the Plot; the bolt's Sprite checks this and removes itself
  from the board if present.
  """

  def __init__(self, curtain, character):
    # The constructor just sets the Marauder's initial horizontal direction.
    super(MarauderDrape, self).__init__(curtain, character)
    self._dx = -1

  def update(self, actions, board, layers, backdrop, things, the_plot):
    # Where are the laser bolts? Only bolts from the player kill a Marauder.
    bolts = np.logical_or.reduce([layers[c] for c in UPWARD_BOLT_CHARS], axis=0)
    hits = bolts & self.curtain                       # Any hits to Marauders?
    np.logical_xor(self.curtain, hits, self.curtain)  # If so, zap the marauder...
    the_plot.add_reward(np.sum(hits)*10)              # ...and supply a reward.
    # Save the identities of marauder-striking bolts in the Plot.
    the_plot['marauder_hitters'] = [chr(c) for c in board[hits]]

    # If no Marauders are left, or if any are sitting on row 10, end the game.
    if (not self.curtain.any()) or self.curtain[10, :].any():
      return the_plot.terminate_episode()  # i.e. return None.

    # We move faster if there are fewer Marauders. The odd divisor causes speed
    # jumps to align on the high sides of multiples of 8; so, speed increases as
    # the number of Marauders decreases to 32 (or 24 etc.), not 31 (or 23 etc.).
    if the_plot.frame % max(1, np.sum(self.curtain)//8.0000001): return
    # If any Marauder reaches either side of the screen, reverse horizontal
    # motion and advance vertically one row.
    if np.any(self.curtain[:, 0] | self.curtain[:, -1]):
      self._dx = -self._dx
      self.curtain[:] = np.roll(self.curtain, shift=1, axis=0)
    self.curtain[:] = np.roll(self.curtain, shift=self._dx, axis=1)


class PlayerSprite(prefab_sprites.MazeWalker):
  """A `Sprite` for our player.

  This `Sprite` simply ties actions to going left and right. In interactive
  settings, the user can also quit.
  """

  def __init__(self, corner, position, character):
    """Simply indicates to the superclass that we can't walk off the board."""
    super(PlayerSprite, self).__init__(
        corner, position, character, impassable='', confined_to_board=True)

  def update(self, actions, board, layers, backdrop, things, the_plot):
    del layers, backdrop, things  # Unused.

    if actions == 0:    # go leftward?
      self._west(board, the_plot)
    elif actions == 1:  # go rightward?
      self._east(board, the_plot)
    elif actions == 4:  # quit?
      the_plot.terminate_episode()


class UpwardLaserBoltSprite(prefab_sprites.MazeWalker):
  """Laser bolts shot from the player toward Marauders."""

  def __init__(self, corner, position, character):
    """Starts the Sprite in a hidden position off of the board."""
    super(UpwardLaserBoltSprite, self).__init__(
        corner, position, character, impassable='')
    self._teleport((-1, -1))

  def update(self, actions, board, layers, backdrop, things, the_plot):
    if self.visible:
      self._fly(board, layers, things, the_plot)
    elif actions == 2:
      self._fire(layers, things, the_plot)

  def _fly(self, board, layers, things, the_plot):
    """Handles the behaviour of visible bolts flying toward Marauders."""
    # Disappear if we've hit a Marauder or a bunker.
    if (self.character in the_plot['bunker_hitters'] or
        self.character in the_plot['marauder_hitters']):
      return self._teleport((-1, -1))
    # Otherwise, northward!
    self._north(board, the_plot)

  def _fire(self, layers, things, the_plot):
    """Launches a new bolt from the player."""
    # We don't fire if the player fired another bolt just now.
    if the_plot.get('last_player_shot') == the_plot.frame: return
    the_plot['last_player_shot'] = the_plot.frame
    # We start just above the player.
    row, col = things['P'].position
    self._teleport((row-1, col))


class DownwardLaserBoltSprite(prefab_sprites.MazeWalker):
  """Laser bolts shot from Marauders toward the player."""

  def __init__(self, corner, position, character):
    """Starts the Sprite in a hidden position off of the board."""
    super(DownwardLaserBoltSprite, self).__init__(
        corner, position, character, impassable='')
    self._teleport((-1, -1))

  def update(self, actions, board, layers, backdrop, things, the_plot):
    if self.visible:
      self._fly(board, layers, things, the_plot)
    else:
      self._fire(layers, the_plot)

  def _fly(self, board, layers, things, the_plot):
    """Handles the behaviour of visible bolts flying toward the player."""
    # Disappear if we've hit a bunker.
    if self.character in the_plot['bunker_hitters']:
      return self._teleport((-1, -1))
    # End the game if we've hit the player.
    if self.position == things['P'].position: the_plot.terminate_episode()
    self._south(board, the_plot)

  def _fire(self, layers, the_plot):
    """Launches a new bolt from a random Marauder."""
    # We don't fire if another Marauder fired a bolt just now.
    if the_plot.get('last_marauder_shot') == the_plot.frame: return
    the_plot['last_marauder_shot'] = the_plot.frame
    # Which Marauder should fire the laser bolt?
    col = np.random.choice(np.nonzero(layers['X'].sum(axis=0))[0])
    row = np.nonzero(layers['X'][:, col])[0][-1] + 1
    # Move ourselves just below that Marauder.
    self._teleport((row, col))


def get_ui():
  repainter = rendering.ObservationCharacterRepainter(LASER_REPAINT_MAPPING)

  # Make a CursesUi to play it with.
  ui = human_ui.CursesUi(
      keys_to_actions={curses.KEY_LEFT: 0, curses.KEY_RIGHT: 1,
                       ' ': 2,   # shoot
                       -1: 3},  # no-op
      repainter=repainter, delay=300,
      colour_fg=COLOURS_FG, colour_bg=COLOURS_BG)
  return ui
