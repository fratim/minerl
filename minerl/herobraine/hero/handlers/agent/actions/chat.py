__author__ = "Sander Schulhoff"
__email__ = "sanderschulhoff@gmail.com"

from minerl.herobraine.hero.handlers.agent.action import Action
import minerl.herobraine.hero.spaces as spaces


# TODO add more command support (things like what commands are allowed) from Malmo
# TODO ensure other agents can send chats, not just first agent (again, check Malmo)
class ChatAction(Action):
    """
    Handler which lets agents send Minecraft chat messages
    Note: this may currently be limited to the
    first agent sending messages (check Malmo for this)
    This can be used to execute MINECRAFT COMMANDS !!!
    Example usage:

    .. code-block:: python
        ChatAction()
    To summon a creeper, use this action dictionary:
    .. code-block:: json
        {"chat": "/summon creeper"}
    """

    def to_string(self):
        return 'chat'

    def to_hero(self, x):
        if x == 0:
            return '' # no items passed on
        elif x == 1:
            return "{} {}".format(self.to_string(), "/give MineRLAgent1 minecraft:planks 1 0") # give 1 plank
        elif x == 2:
            return "{} {}".format(self.to_string(), "/give MineRLAgent1 minecraft:planks 10 0") # give 10 planks
        else:
            raise NotImplementedError

    # TODO test that this implementation actually works...
    # TODO add relative position between agents to observation space

    def xml_template(self) -> str:
        return str("<ChatCommands> </ChatCommands>")

    def __init__(self):
        import warnings
        warnings.warn("Currently, only agent_0 can use this handler")

        self._command = 'chat'
        super().__init__(self.command, spaces.Discrete(3))

    def from_universal(self, x):
        return []