from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

def sanitize_messages(messages, actor_name=None):
    """
    Sanitize messages to ensure stay within provider limits:
    1. Only SystemMessage, HumanMessage, AIMessage allowed.
    2. No empty content.
    3. Strict User/Assistant alternation.
    4. Merge consecutive same-role messages.
    5. Strip name attributes (problematic for some providers).
    """
    if not messages:
        return []

    # Final list of LangChain message objects
    processed = []
    
    for msg in messages:
        content = msg.content if hasattr(msg, 'content') else str(msg)
        if not content or str(content).strip() == "":
            continue

        if isinstance(msg, SystemMessage):
            processed.append(SystemMessage(content=content))
        elif isinstance(msg, AIMessage):
            # If actor_name is set, only messages matching that name stay as AIMessage
            # Others become HumanMessage to ensure the API sees a User -> Assistant flow
            if actor_name and hasattr(msg, 'name') and msg.name != actor_name:
                processed.append(HumanMessage(content=f"[{msg.name or 'Assistant'}] {content}"))
            else:
                processed.append(AIMessage(content=content))
        else:
            processed.append(HumanMessage(content=content))

    if not processed:
        return []

    # Ensure strict User/Assistant alternation after system message
    final = []
    system_messages = [m for m in processed if isinstance(m, SystemMessage)]
    other_messages = [m for m in processed if not isinstance(m, SystemMessage)]
    
    # Merge same-role system messages
    if system_messages:
        merged_system = system_messages[0]
        for sm in system_messages[1:]:
            merged_system.content += "\n\n" + sm.content
        final.append(merged_system)
    
    # Merge and alternate other messages
    temp_other = []
    for msg in other_messages:
        if not temp_other:
            temp_other.append(msg)
            continue
        
        last = temp_other[-1]
        if type(last) == type(msg):
            last.content += "\n\n" + msg.content
        else:
            temp_other.append(msg)
    
    for i, msg in enumerate(temp_other):
        if not final or isinstance(final[-1], SystemMessage):
            # First non-system message MUST be HumanMessage (User)
            if isinstance(msg, AIMessage):
                final.append(HumanMessage(content=f"[Answer] {msg.content}"))
            else:
                final.append(msg)
        else:
            last = final[-1]
            if type(last) == type(msg):
                last.content += "\n\n" + msg.content
            else:
                final.append(msg)
    
    # Final pass: Strip name attributes as some providers crash on them
    for m in final:
        if hasattr(m, 'name'):
            m.name = None
                
    return final
