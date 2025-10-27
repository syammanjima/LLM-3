"""
Multi-Agent System for Coding, Testing, and Debugging using LangGraph
=====================================================================
File: multi_agent_system.py
"""

import asyncio
import json
import subprocess
import tempfile
import os
from typing import Dict, List, Any, Optional, Annotated, TypedDict
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool


# ============================================================================
# STATE DEFINITIONS
# ============================================================================

@dataclass
class CodeProject:
    """Represents the current state of a coding project."""
    requirement: str = ""
    code: str = ""
    tests: str = ""
    test_results: str = ""
    bugs_found: List[str] = field(default_factory=list)
    fixes_applied: List[str] = field(default_factory=list)
    is_complete: bool = False
    iteration_count: int = 0


class AgentState(TypedDict):
    """Global state shared between all agents."""
    messages: Annotated[list, add_messages]
    project: CodeProject
    current_agent: str
    next_action: str


# ============================================================================
# UTILITY TOOLS
# ============================================================================

@tool
def execute_code(code: str, test_code: str = "") -> Dict[str, Any]:
    """Execute Python code and optional tests in an isolated environment."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        code_file = Path(tmp_dir) / "main.py"
        code_file.write_text(code)
        
        test_file = Path(tmp_dir) / "test_main.py"
        if test_code:
            test_file.write_text(test_code)
        
        results = {"stdout": "", "stderr": "", "return_code": 0}
        
        try:
            result = subprocess.run(
                ["python", str(code_file)],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=tmp_dir
            )
            results.update({
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            })
            
            if test_code:
                test_result = subprocess.run(
                    ["python", "-m", "pytest", str(test_file), "-v"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=tmp_dir
                )
                results["test_output"] = test_result.stdout
                results["test_errors"] = test_result.stderr
                results["test_return_code"] = test_result.returncode
                
        except subprocess.TimeoutExpired:
            results["stderr"] = "Code execution timed out (10 seconds)"
            results["return_code"] = -1
        except Exception as e:
            results["stderr"] = f"Execution error: {str(e)}"
            results["return_code"] = -1
            
        return results


@tool 
def static_analysis(code: str) -> Dict[str, Any]:
    """Perform static analysis on code to find potential issues."""
    issues = []
    
    lines = code.split('\n')
    for i, line in enumerate(lines, 1):
        if 'TODO' in line or 'FIXME' in line:
            issues.append(f"Line {i}: Unfinished code comment found")
        if line.strip().endswith('pass') and len(line.strip()) == 4:
            issues.append(f"Line {i}: Empty implementation (pass statement)")
        if 'print(' in line and 'debug' in line.lower():
            issues.append(f"Line {i}: Debug print statement found")
    
    try:
        compile(code, '<string>', 'exec')
    except SyntaxError as e:
        issues.append(f"Syntax Error: {e.msg} at line {e.lineno}")
    
    return {
        "issues_found": len(issues),
        "issues": issues,
        "analysis_complete": True
    }


# ============================================================================
# AGENT IMPLEMENTATIONS
# ============================================================================

class CodingAgent:
    """Agent responsible for writing and improving code."""
    
    def __init__(self, llm):
        self.llm = llm
        self.name = "CodingAgent"
    
    async def process(self, state: AgentState) -> AgentState:
        """Generate or improve code based on requirements."""
        project = state["project"]
        
        system_prompt = """You are an expert Python developer. Your job is to write clean, 
        efficient, and well-documented code based on requirements.
        
        Guidelines:
        - Write production-ready code with proper error handling
        - Include docstrings and comments where helpful
        - Follow PEP 8 style guidelines
        - Consider edge cases and input validation
        - Make code modular and reusable
        - Use type hints for function parameters and return values
        """
        
        if project.code == "":
            user_prompt = f"""
            Write Python code for the following requirement:
            
            {project.requirement}
            
            Provide ONLY the Python code without any explanations or markdown.
            """
        else:
            user_prompt = f"""
            Improve the following code based on these issues:
            
            Bugs found: {project.bugs_found if project.bugs_found else "None"}
            Test results: {project.test_results}
            
            Current code:
            ```python
            {project.code}
            ```
            
            Provide the improved Python code without explanations.
            """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        
        new_code = response.content.strip()
        if new_code.startswith('```python'):
            new_code = new_code.split('```python')[1].split('```')[0].strip()
        elif new_code.startswith('```'):
            new_code = new_code.split('```')[1].split('```')[0].strip()
        
        project.code = new_code
        project.iteration_count += 1
        
        state["messages"].append(AIMessage(
            content=f"[{self.name}] Generated/improved code (iteration {project.iteration_count})"
        ))
        state["current_agent"] = self.name
        state["next_action"] = "test"
        
        return state


class TestingAgent:
    """Agent responsible for creating and running tests."""
    
    def __init__(self, llm):
        self.llm = llm
        self.name = "TestingAgent"
    
    async def process(self, state: AgentState) -> AgentState:
        """Generate comprehensive tests for the code."""
        project = state["project"]
        
        system_prompt = """You are an expert test engineer. Write comprehensive unit tests 
        using pytest for the given code.
        
        Guidelines:
        - Test all functions and methods thoroughly
        - Include edge cases and boundary conditions
        - Test error conditions and exceptions
        - Use descriptive test names
        - Test both positive and negative scenarios
        """
        
        user_prompt = f"""
        Write comprehensive pytest tests for this code:
        
        ```python
        {project.code}
        ```
        
        Requirement context: {project.requirement}
        
        Provide ONLY the test code using pytest without explanations.
        Include all necessary imports at the top.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        
        test_code = response.content.strip()
        if test_code.startswith('```python'):
            test_code = test_code.split('```python')[1].split('```')[0].strip()
        elif test_code.startswith('```'):
            test_code = test_code.split('```')[1].split('```')[0].strip()
        
        project.tests = test_code
        
        # Execute tests using invoke method (FIXED)
        execution_result = execute_code.invoke({"code": project.code, "test_code": test_code})
        project.test_results = json.dumps(execution_result, indent=2)
        
        state["messages"].append(AIMessage(
            content=f"[{self.name}] Created and executed tests"
        ))
        
        state["current_agent"] = self.name
        
        if execution_result.get("test_return_code", 0) != 0 or execution_result.get("return_code", 0) != 0:
            state["next_action"] = "debug"
        else:
            state["next_action"] = "review"
        
        return state


class DebuggingAgent:
    """Agent responsible for finding and fixing bugs."""
    
    def __init__(self, llm):
        self.llm = llm
        self.name = "DebuggingAgent"
    
    async def process(self, state: AgentState) -> AgentState:
        """Analyze test results and identify bugs."""
        project = state["project"]
        
        # Perform static analysis using invoke method (FIXED)
        static_result = static_analysis.invoke({"code": project.code})
        
        system_prompt = """You are an expert debugger. Analyze code, test results, and 
        static analysis to identify bugs and issues.
        
        Guidelines:
        - Identify root causes of failures
        - Provide specific, actionable bug descriptions
        - Consider both runtime errors and logic errors
        - Be concise but clear
        """
        
        user_prompt = f"""
        Analyze this code and its test results to find bugs:
        
        Code:
        ```python
        {project.code}
        ```
        
        Test Results:
        {project.test_results}
        
        Static Analysis:
        {json.dumps(static_result, indent=2)}
        
        Identify specific bugs and issues. List each bug on a new line.
        Provide ONLY the bug descriptions, no code fixes.
        If no bugs found, respond with "No bugs found".
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        
        if "no bugs found" in response.content.lower():
            bugs = []
        else:
            bugs = [bug.strip() for bug in response.content.split('\n') 
                   if bug.strip() and not bug.startswith('#') and len(bug.strip()) > 5]
        
        project.bugs_found = bugs
        
        state["messages"].append(AIMessage(
            content=f"[{self.name}] Found {len(bugs)} bug(s)"
        ))
        
        state["current_agent"] = self.name
        state["next_action"] = "fix" if bugs else "review"
        
        return state


class CoordinatorAgent:
    """Agent that coordinates the workflow and makes routing decisions."""
    
    def __init__(self):
        self.name = "CoordinatorAgent"
        self.max_iterations = 3
    
    async def process(self, state: AgentState) -> AgentState:
        """Coordinate the workflow and decide next steps."""
        project = state["project"]
        
        try:
            test_results_dict = json.loads(project.test_results) if project.test_results else {}
            tests_passed = test_results_dict.get("test_return_code", 1) == 0
        except:
            tests_passed = False
        
        if project.iteration_count >= self.max_iterations:
            project.is_complete = True
            state["next_action"] = "complete"
        elif (project.code and project.tests and not project.bugs_found and tests_passed):
            project.is_complete = True
            state["next_action"] = "complete"
        elif project.bugs_found and state["next_action"] == "fix":
            state["next_action"] = "code"
        elif state["next_action"] == "review":
            if not project.bugs_found and tests_passed:
                project.is_complete = True
                state["next_action"] = "complete"
            else:
                state["next_action"] = "code"
        
        state["messages"].append(AIMessage(
            content=f"[{self.name}] Decision: {state['next_action']} (iteration {project.iteration_count})"
        ))
        state["current_agent"] = self.name
        
        return state


# ============================================================================
# MAIN SYSTEM CLASS
# ============================================================================

class MultiAgentCodingSystem:
    """Main system that orchestrates the multi-agent workflow."""
    
    def __init__(self, anthropic_api_key: str, model: str = "claude-3-5-haiku-20241022"):
        """
        Initialize the multi-agent system with Claude.
        
        Args:
            anthropic_api_key: Anthropic API key
            model: Claude model to use (default: claude-3-5-haiku-20241022)
        """
        self.llm = ChatAnthropic(
            model=model,
            api_key=anthropic_api_key,
            temperature=0.1,
            max_tokens=4096
        )
        
        self.coding_agent = CodingAgent(self.llm)
        self.testing_agent = TestingAgent(self.llm)
        self.debugging_agent = DebuggingAgent(self.llm)
        self.coordinator_agent = CoordinatorAgent()
        
        self.workflow = self._build_workflow()
        
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("coordinator", self.coordinator_agent.process)
        workflow.add_node("coding", self.coding_agent.process)
        workflow.add_node("testing", self.testing_agent.process)
        workflow.add_node("debugging", self.debugging_agent.process)
        
        workflow.add_edge(START, "coordinator")
        
        def route_next_action(state: AgentState):
            action = state["next_action"]
            if action == "code":
                return "coding"
            elif action == "test":
                return "testing"
            elif action == "debug":
                return "debugging"
            elif action == "complete":
                return END
            else:
                return "coordinator"
        
        workflow.add_conditional_edges(
            "coordinator",
            route_next_action,
            {
                "coding": "coding",
                "testing": "testing", 
                "debugging": "debugging",
                END: END
            }
        )
        
        workflow.add_edge("coding", "coordinator")
        workflow.add_edge("testing", "coordinator")
        workflow.add_edge("debugging", "coordinator")
        
        return workflow.compile(checkpointer=MemorySaver())
    
    async def develop_code(self, requirement: str) -> Dict[str, Any]:
        """Main method to develop code based on requirements."""
        initial_state: AgentState = {
            "messages": [HumanMessage(content=f"Develop code for: {requirement}")],
            "project": CodeProject(requirement=requirement),
            "current_agent": "",
            "next_action": "code"
        }
        
        config = {"configurable": {"thread_id": "coding_session"}}
        
        result = await self.workflow.ainvoke(initial_state, config)
        
        return {
            "requirement": requirement,
            "final_code": result["project"].code,
            "tests": result["project"].tests,
            "test_results": result["project"].test_results,
            "bugs_found": result["project"].bugs_found,
            "iterations": result["project"].iteration_count,
            "completed": result["project"].is_complete,
            "conversation_history": [msg.content for msg in result["messages"]]
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Example usage of the multi-agent coding system with Claude."""
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        print("❌ Error: ANTHROPIC_API_KEY not found!")
        print("Please set it using: set ANTHROPIC_API_KEY='your-key'")
        return
    
    print("🚀 Initializing Multi-Agent Coding System (Claude Edition)...\n")
    system = MultiAgentCodingSystem(api_key)
    
    requirement = """
    Create a Python function that calculates the factorial of a number.
    The function should handle edge cases like negative numbers and zero.
    Include proper error handling and return appropriate error messages.
    """
    
    print(f"📝 Requirement:\n{requirement}")
    print("\n⏳ Agents are working... This may take 1-2 minutes.\n")
    
    try:
        result = await system.develop_code(requirement)
        
        print("=" * 70)
        print("✅ DEVELOPMENT COMPLETED!")
        print("=" * 70)
        
        print(f"\n📊 Statistics:")
        print(f"   • Iterations: {result['iterations']}")
        print(f"   • Status: {'✅ Complete' if result['completed'] else '⚠️ Incomplete'}")
        print(f"   • Bugs Found: {len(result['bugs_found'])}")
        
        print(f"\n💻 FINAL CODE:")
        print("-" * 70)
        print(result['final_code'])
        
        print(f"\n🧪 TESTS:")
        print("-" * 70)
        print(result['tests'])
        
        print(f"\n📋 TEST RESULTS:")
        print("-" * 70)
        print(result['test_results'])
        
        if result['bugs_found']:
            print(f"\n🐛 BUGS FOUND:")
            print("-" * 70)
            for i, bug in enumerate(result['bugs_found'], 1):
                print(f"{i}. {bug}")
        
        print("\n" + "=" * 70)
        print("🎉 Process Complete!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error during development: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
