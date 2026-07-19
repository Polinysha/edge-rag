_parent_blocks: dict[str, dict] = {}


def store_parent_block(parent_id: str, parent_block: dict) -> None:

    _parent_blocks[parent_id] = parent_block


def get_parent_block(parent_id: str) -> dict | None:

    return _parent_blocks.get(parent_id)


def get_parent_blocks(parent_ids: list[str]) -> list[dict]:

    seen = set()
    result = []
    for parent_id in parent_ids:
        if parent_id not in seen:
            block = get_parent_block(parent_id)
            if block:
                result.append(block)
                seen.add(parent_id)
    return result


def clear_parent_store() -> None:
    _parent_blocks.clear()


def get_all_parent_ids() -> list[str]:
    return list(_parent_blocks.keys())


def get_store_size() -> int:
    return len(_parent_blocks)
