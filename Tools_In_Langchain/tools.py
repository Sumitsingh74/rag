# from langchain_community.tools import DuckDuckGoSearchRun
#
# search_tool = DuckDuckGoSearchRun()
#
# results = search_tool.invoke('top news from india today')
#
# print(results)

from langchain_community.tools import ShellTool

shell_tool = ShellTool()

results = shell_tool.invoke('whoami')

print(results)