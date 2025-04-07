import openai
from colorama import Fore, init

# Initialize colorama for colored terminal output
init(autoreset=True)

class Agent:
    def __init__(self, name, model, tools=None, system_prompt=""):
        self.name = name
        self.model = model  # e.g. "gpt-4"
        self.messages = []
        self.tools = tools if tools is not None else []
        self.tools_schemas = self.get_openai_tools_schema() if self.tools else None
        self.system_prompt = system_prompt
        if self.system_prompt and not self.messages:
            self.handle_messages_history("system", self.system_prompt)

    def invoke(self, message):
        print(Fore.GREEN + f"\nCalling Agent: {self.name}")
        self.handle_messages_history("user", message)
        result = self.execute()
        return result

    def execute(self):
        response_message = self.call_llm()
        response_content = response_message.get("content", "")
        tool_calls = response_message.get("tool_calls", [])
        if tool_calls:
            try:
                response_content = self.run_tools(tool_calls)
            except Exception as e:
                print(Fore.RED + f"\nError: {e}\n")
        return response_content

    def run_tools(self, tool_calls):
        for tool_call in tool_calls:
            self.execute_tool(tool_call)
        # After running a tool, re-call the LLM to incorporate the tool’s output.
        response_content = self.execute()
        return response_content

    def execute_tool(self, tool_call):
        function_name = tool_call.get("name")
        # Find the tool function by name
        func = next((func for func in self.tools if func.__name__ == function_name), None)
        if not func:
            return f"Error: Function {function_name} not found. Available functions: {[func.__name__ for func in self.tools]}"
        try:
            print(Fore.GREEN + f"\nCalling Tool: {function_name}")
            print(Fore.GREEN + f"Arguments: {tool_call.get('arguments')}\n")
            # Evaluate the arguments string into a dict (ensure safe usage in your context)
            func_args = eval(tool_call.get("arguments", "{}"))
            output = func(**func_args).run()
            tool_message = {"name": function_name, "tool_call_id": tool_call.get("id")}
            self.handle_messages_history("tool", output, tool_output=tool_message)
            return output
        except Exception as e:
            print("Error: ", str(e))
            return "Error: " + str(e)

    def call_llm(self):
        # Prepare function definitions if any tools are available (for OpenAI function calling)
        functions = None
        if self.tools and self.tools_schemas:
            functions = [tool_schema for tool_schema in self.tools_schemas]
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
            temperature=0.1,
            functions=functions,
        )
        message = response.choices[0].message

        # Process function calls if present
        if "function_call" in message and message["function_call"]:
            # Wrap the function call into a list for compatibility
            message["tool_calls"] = [message["function_call"]]
        else:
            message["tool_calls"] = []

        self.handle_messages_history("assistant", message.get("content", ""), tool_calls=message.get("tool_calls", []))
        return message

    def get_openai_tools_schema(self):
        return [
            {"type": "function", "function": tool.openai_schema} for tool in self.tools
        ]

    def reset(self):
        self.messages = []
        if self.system_prompt:
            self.handle_messages_history("system", self.system_prompt)

    def handle_messages_history(self, role, content, tool_calls=None, tool_output=None):
        message = {"role": role, "content": content}
        if tool_calls:
            message["tool_calls"] = self.parse_tool_calls(tool_calls)
        if tool_output:
            message["name"] = tool_output["name"]
            message["tool_call_id"] = tool_output["tool_call_id"]
        self.messages.append(message)

    def parse_tool_calls(self, calls):
        parsed_calls = []
        for call in calls:
            parsed_call = {
                "name": call.get("name", ""),
                "arguments": call.get("arguments", "{}"),
                "id": call.get("id", ""),
                "type": call.get("type", ""),
            }
            parsed_calls.append(parsed_call)
        return parsed_calls
