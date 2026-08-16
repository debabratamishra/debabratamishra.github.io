# Custom liquid tag for rendering MathJax equations as code-styled snippets.
# Usage in posts: {% equation %}$$E = mc^2$${% endequation %}
#
# This tag wraps $$...$$ in a div with class MathJax_Display so:
# 1. MathJax can still render the equation
# 2. The existing SCSS styling (.MathJax_Display in _base.scss) applies code-styled appearance
# 3. fitEquations() can scale the equation to fit the viewport and add ``` fences

module Jekyll
  class EquationTag < Liquid::Tag
    def render(_context)
      body = @content.strip

      # Look for $$...$$ pattern
      if body.match(/\$\$(.+?)\$\$/m)
        math_content = $1
        # Wrap in a div with MathJax_Display class.
        # MathJax will process the $$...$$ and render formatted math.
        # The existing CSS (.MathJax_Display in _base.scss) provides code-styled appearance
        # (monospace, padding, border, shadow). fitEquations() will add ``` fences and scale.
        "<div class=\"MathJax_Display\">\n" + math_content + "\n</div>"
      else
        # If no $$ found, wrap in code fence div for CSS styling
        "<div class=\"MathJax_Display\">\n" + body + "\n</div>"
      end
    end
  end
end

# Register the liquid tag - this must be at the top level for Jekyll to pick it up
Liquid::Template.register_tag('equation', Jekyll::EquationTag)