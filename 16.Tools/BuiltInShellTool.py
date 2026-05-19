from langchain_community.tools import ShellTool

search_tool = ShellTool()

result = search_tool.invoke('ls')

print(result)

print(search_tool.name)
print(search_tool.description)
print(search_tool.args_schema)
print(search_tool.args)