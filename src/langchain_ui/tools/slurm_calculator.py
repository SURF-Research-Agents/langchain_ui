import os
from langchain_surf.tools.hpc_tools import tool

slurm_data = {
    "url": "https://slurm.snellius.surf.nl",
    "api_ver": "v0.0.43",
    "user_name": os.getenv('SLURM_USER'),
    "slurm_jwt": os.getenv('SLURM_JWT'),
}

hpc_opt = {
    'slurm_data': slurm_data,
}

@tool(hpc=hpc_opt)
def slurm_calculator(expression):
    """Evaluate the mathematical expression and return the result."""
    
    import ast
    import operator
    from collections.abc import Callable

    OPERATORS: dict[type[ast.operator], Callable] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
    }

    def _eval_expr(node: ast.AST) -> float:
        """Evaluate an AST node recursively."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, int or float):
                return float(node.value)
            error_msg = f"Unsupported constant type: {type(node.value).__name__}"
            raise TypeError(error_msg)
        if isinstance(node, ast.Constant):  # For backwards compatibility
            if isinstance(node.n, int or float):
                return float(node.n)
            error_msg = f"Unsupported number type: {type(node.n).__name__}"
            raise TypeError(error_msg)

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in OPERATORS:
                error_msg = f"Unsupported binary operator: {op_type.__name__}"
                raise TypeError(error_msg)

            left = _eval_expr(node.left)
            right = _eval_expr(node.right)
            return OPERATORS[op_type](left, right)

        error_msg = f"Unsupported operation or expression type: {type(node).__name__}"
        raise TypeError(error_msg)
    
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_expr(tree.body)

        formatted_result = f"{float(result):.6f}".rstrip("0").rstrip(".")
        return {'result': formatted_result}

    except ZeroDivisionError:
        error_message = "Error: Division by zero"
        return {'error': error_message, 'input': expression}

    except (SyntaxError, TypeError, KeyError, ValueError, AttributeError, OverflowError) as e:
        error_message = f"Invalid expression: {e!s}"
        return {'error': error_message, 'input': expression}