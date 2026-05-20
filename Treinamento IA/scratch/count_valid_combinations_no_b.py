def count_valid_no_rule_b():
    valid_count = 0
    valid_keys = []
    
    # States
    pinky_states = [0, 1, 2, 3]
    pr_spreads = [0, 1]
    ring_states = [0, 1, 2, 3]
    rm_spreads = [0, 1]
    middle_states = [0, 1, 2, 3]
    mi_spreads = [0, 1]
    index_states = [0, 1, 2, 3]
    it_spreads = [0, 1]
    thumb_opps = [0, 1]
    thumb_states = [0, 2, 3]
    
    for pinky_s in pinky_states:
        for pr_sp in pr_spreads:
            for ring_s in ring_states:
                # Rule A: Spread constraint
                # Pinky-Ring spread:
                if pr_sp == 1 and (pinky_s > 1 or ring_s > 1):
                    continue
                
                for rm_sp in rm_spreads:
                    for middle_s in middle_states:
                        if rm_sp == 1 and (ring_s > 1 or middle_s > 1):
                            continue
                        
                        for mi_sp in mi_spreads:
                            for index_s in index_states:
                                if mi_sp == 1 and (middle_s > 1 or index_s > 1):
                                    continue
                                
                                for it_sp in it_spreads:
                                    for thumb_opp in thumb_opps:
                                        for thumb_s in thumb_states:
                                            if it_sp == 1 and (index_s > 1 or thumb_s > 1):
                                                continue
                                            
                                            key = f"{pinky_s}{pr_sp}{ring_s}{rm_sp}{middle_s}{mi_sp}{index_s}{it_sp}{thumb_opp}{thumb_s}"
                                            valid_keys.append(key)
                                            valid_count += 1
                                            
    print(f"Total combinations without Rule B: {valid_count}")

count_valid_no_rule_b()
