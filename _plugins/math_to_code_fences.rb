# Converts $$ display math to code fences for code-styled appearance.
# This ensures equations render as code-styled snippets that persist in static generation,
# rather than relying on client-side JavaScript (fitEquations) that doesn't persist.
#
# This plugin uses Jekyll's post_read hook to convert $$...$$ to fenced code blocks
# BEFORE Kramdown processes the Markdown, so the conversion persists in the final HTML.

module Jekyll
  module MathToCodeFences
    # Convert $$...$$ to fenced code blocks
    def self.convert_display_math(content)
      # Match $$ followed by any content (including newlines) until the closing $$
      # Use m flag for multiline matching, and .+? for non-greedy match
      regex = /\$\$(.+?)\$\$\n?/m

      content.gsub(regex) do |match|
        # Extract the math content between the $$ delimiters
        math_content = $1
        # Wrap in code fences
        "`" + "`" + "`" + "\n" + math_content + "\n" + "`" + "`" + "`"
      end
    end
  end
end

# Hook into post_read to convert $$ to code fences BEFORE Kramdown processes
Jekyll::Hooks.register :documents, :post_read do |doc|
  next unless doc.extname == '.md' || doc.extname == '.markdown'

  content = doc.content.to_s
  # Only process if $$ is present and the document hasn't been processed yet
  next unless content.include?('$$')

  converted = MathToCodeFences.convert_display_math(content)

  # Only update if changes were made
  if converted != content
    doc.content = converted
  end
end